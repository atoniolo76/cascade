"""Kernel display names for Perfetto slices; the twin of web/src/domain/kernelName.ts.

Both must return the same string for every input, so keep changes to one mirrored in the other.
"""

import itertools
import re

_ANON = "(anonymous namespace)"
_WRAPPERS = frozenset({"cutlass::Kernel", "cutlass::Kernel2", "cutlass::device_kernel"})

# JavaScript's String.prototype.trim set; str.strip() would also take \x1c-\x1f and \x85, and
# would keep ﻿.
_JS_WHITESPACE = "\t\n\v\f\r                  　﻿"

_NUMBER = re.compile(r"n?[0-9]+")
_DIGITS = frozenset("0123456789")
_SEQ_ID = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_LOWER = frozenset("abcdefghijklmnopqrstuvwxyz")
_DTOR = frozenset("012345")
_LITERAL = frozenset("0123456789abcdefn_.")


class _Bail(Exception):
    pass


class _Itanium:
    """Just enough of the Itanium C++ ABI grammar to read the function's qualified name.

    Substitutions are skipped, not resolved: a function's own prefix never starts with one.
    The `in "..."` checks without an `!= ""` guard match the empty string on purpose, the way
    JavaScript's String.includes("") does in the twin.
    """

    def __init__(self, s: str) -> None:
        self.s = s
        self.pos = 0
        self.parts: list[str] = []
        self.first_arg: str | None = None

    def _peek(self, offset: int = 0) -> str:
        i = self.pos + offset
        return self.s[i] if i < len(self.s) else ""

    def _take(self, ch: str) -> None:
        if self._peek() != ch:
            raise _Bail
        self.pos += 1

    def _number(self) -> int:
        m = _NUMBER.match(self.s, self.pos)
        if m is None:
            raise _Bail
        self.pos = m.end()
        return int(m.group().replace("n", "-"))

    def _seq_id(self) -> None:
        while self._peek() in _SEQ_ID:
            self.pos += 1
        self._take("_")

    def _source_name(self) -> str:
        length = self._number()
        if length <= 0 or self.pos + length > len(self.s):
            raise _Bail
        ident = self.s[self.pos : self.pos + length]
        self.pos += length
        return _ANON if ident.startswith("_GLOBAL__N") else ident

    def encoding(self) -> None:
        self._take("_")
        self._take("Z")
        self._name(self.parts)

    def _name(self, into: list[str]) -> None:
        c = self._peek()
        if c == "N":
            self._nested(into)
        elif c == "Z":
            self._local(into)
        else:
            if c == "S" and self._peek(1) == "t":
                self.pos += 2
                into.append("std")
            self._unqualified(into)
            if self._peek() == "I":
                self._template_args(into is self.parts)

    def _nested(self, into: list[str]) -> None:
        self._take("N")
        while self._peek() != "" and self._peek() in "rVKRO":
            self.pos += 1
        while self._peek() != "E":
            c = self._peek()
            if c == "":
                raise _Bail
            if c == "I":
                self._template_args(into is self.parts)
            elif c == "S":
                if self._peek(1) == "t" or self._peek(1) in "absiod":
                    self.pos += 2
                    into.append("std")
                else:
                    self.pos += 1
                    self._seq_id()
            elif c == "T":
                self.pos += 1
                self._seq_id()
            elif c == "M":
                self.pos += 1
            else:
                self._unqualified(into)
        self.pos += 1

    def _unqualified(self, into: list[str]) -> None:
        if self._peek() == "L":
            self.pos += 1
        d = self._peek()
        if d in _DIGITS:
            into.append(self._source_name())
        elif d in ("C", "D"):
            if d == "D" and self._peek(1) not in _DTOR:
                raise _Bail
            cls = into[-1] if into else ""
            into.append(f"~{cls}" if d == "D" else cls)
            self.pos += 2
        elif d == "U":
            self.pos += 1
            if self._peek() == "t":
                self.pos += 1
                if self._peek() != "_":
                    self._number()
                self._take("_")
                into.append("{unnamed type}")
            elif self._peek() == "l":
                self.pos += 1
                while self._peek() != "E":
                    self._type()
                self.pos += 1
                if self._peek() != "_":
                    self._number()
                self._take("_")
                into.append("{lambda}")
            else:
                raise _Bail
        elif into and d in _LOWER and self._peek(1) in _LOWER:
            op = self.s[self.pos : self.pos + 2]
            self.pos += 2
            into.append("operator()" if op == "cl" else f"operator {op}")
        else:
            raise _Bail
        while self._peek() == "B":
            self.pos += 1
            self._source_name()

    def _local(self, into: list[str]) -> None:
        self._take("Z")
        fn: list[str] = []
        self._name(fn)
        while self._peek() != "E":
            self._type()
        self.pos += 1
        into.extend(fn)
        if self._peek() == "s":
            self.pos += 1
        else:
            self._name(into)
        if self._peek() == "_":
            self.pos += 1
            if self._peek() == "_":
                self.pos += 1
                self._number()
                self._take("_")
            else:
                self._number()

    def _template_args(self, capture: bool) -> None:
        self._take("I")
        first = True
        while self._peek() != "E":
            if self._peek() == "":
                raise _Bail
            start = self.pos
            self._template_arg()
            if capture and first:
                self.first_arg = (
                    _Itanium(self.s[start : self.pos]).leading_source_name()
                    if self.s[start : start + 1] in _DIGITS
                    else None
                )
            first = False
        self.pos += 1

    def leading_source_name(self) -> str | None:
        try:
            return self._source_name()
        except _Bail:
            return None

    def _template_arg(self) -> None:
        c = self._peek()
        if c == "L":
            self._expr_primary()
        elif c == "J":
            self.pos += 1
            while self._peek() != "E":
                self._template_arg()
            self.pos += 1
        elif c == "X":
            raise _Bail
        else:
            self._type()

    def _expr_primary(self) -> None:
        self._take("L")
        if self._peek() == "_" and self._peek(1) == "Z":
            self.pos += 2
            self._name([])
            while self._peek() != "E":
                self._type()
        else:
            self._type()
            while self._peek() in _LITERAL:
                self.pos += 1
        self._take("E")

    def _type(self) -> None:
        c = self._peek()
        if c != "" and c in "vwbcahstijlmxynofdegz":
            self.pos += 1
            return
        match c:
            case "P" | "R" | "O" | "K" | "V" | "r" | "C" | "G":
                self.pos += 1
                self._type()
            case "U":
                self.pos += 1
                self._source_name()
                if self._peek() == "I":
                    self._template_args(False)
                self._type()
            case "D":
                self._builtin_d()
            case "N":
                self._nested([])
            case "Z":
                self._local([])
            case "S":
                self.pos += 1
                if self._peek() == "t":
                    self.pos += 1
                    self._unqualified([])
                elif self._peek() != "" and self._peek() in "absiod":
                    self.pos += 1
                else:
                    self._seq_id()
                if self._peek() == "I":
                    self._template_args(False)
            case "T":
                self.pos += 1
                self._seq_id()
                if self._peek() == "I":
                    self._template_args(False)
            case "F":
                self.pos += 1
                if self._peek() == "Y":
                    self.pos += 1
                while self._peek() != "E":
                    if self._peek() in ("R", "O") and self._peek(1) == "E":
                        self.pos += 1
                    else:
                        self._type()
                self.pos += 1
            case "A":
                self.pos += 1
                if self._peek() != "_":
                    self._number()
                self._take("_")
                self._type()
            case "M":
                self.pos += 1
                self._type()
                self._type()
            case _ if c in _DIGITS:
                self._source_name()
                if self._peek() == "I":
                    self._template_args(False)
            case _:
                raise _Bail

    def _builtin_d(self) -> None:
        self.pos += 1
        c = self._peek()
        self.pos += 1
        if c != "" and c in "dfehiusacn":
            return
        if c == "p":
            self._type()
            return
        if c == "F":
            self._number()
            self._take("_")
            return
        if c == "v":
            self._number()
            self._take("_")
            self._type()
            return
        raise _Bail


def _from_parts(parts: list[str], full: str, first_arg: str | None = None) -> tuple[str, str]:
    named = [p for p in parts if p != _ANON]
    if not named:
        return "", full
    base = named[-1]
    scope = "".join(f"{p}::" for p in named[:-1])
    if f"{scope}{base}" in _WRAPPERS and first_arg is not None:
        return "", first_arg
    return scope, base


def _demangle(full: str) -> tuple[str, str]:
    # The twin indexes UTF-16 code units, so <source-name> lengths count an astral character
    # as two; read the name as code units and rejoin surrogate pairs on the way out.
    units = "".join(
        chr(int.from_bytes(pair, "little"))
        for pair in itertools.batched(full.encode("utf-16-le", "surrogatepass"), 2)
    )
    p = _Itanium(units)
    try:
        p.encoding()
    except _Bail:
        pass
    scope, base = _from_parts(p.parts, units, p.first_arg) if p.parts else ("", units)
    return _from_units(scope), _from_units(base)


def _from_units(s: str) -> str:
    return s.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def _from_text(full: str) -> tuple[str, str]:
    """Already-demangled text such as `void ns::k<4, float>(float*, int)`: keep `ns::k`."""
    text = full.replace(_ANON, "\0")
    depth = 0
    out = []
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            if ch == "(":
                break
            out.append(ch)
    name = "".join(out).strip(_JS_WHITESPACE).split(" ")[-1]
    parts = [_ANON if p == "\0" else p for p in name.split("::") if p]
    return _from_parts(parts, full)


def _simplify(name: str) -> tuple[str, str]:
    full = name.strip(_JS_WHITESPACE)
    if full.startswith("_Z"):
        return _demangle(full)
    if "::" in full or "(" in full:
        return _from_text(full)
    return "", full


def kernel_display_name(name: str) -> str:
    scope, base = _simplify(name)
    return f"{scope}{base}"
