/** A kernel name cut down to what a person scans for: `scope` is the namespaces with a trailing `::`, `base` the function, `full` the name as recorded. */
export interface KernelName {
  scope: string;
  base: string;
  full: string;
}

const ANON = "(anonymous namespace)";
/** Generic launchers whose first template argument is the kernel that actually runs; they read as that argument. */
const WRAPPERS = new Set(["cutlass::Kernel", "cutlass::Kernel2", "cutlass::device_kernel"]);

class Bail extends Error {}

/**
 * Just enough of the Itanium C++ ABI grammar to read the function's qualified name and skip
 * everything else (template arguments, parameter types). Substitutions are skipped, not resolved:
 * a function's own prefix never starts with one, so the name never needs them.
 */
class Itanium {
  pos = 0;
  readonly parts: string[] = [];
  firstArg: string | null = null;

  private readonly s: string;

  constructor(s: string) {
    this.s = s;
  }

  private peek(offset = 0): string {
    return this.s[this.pos + offset] ?? "";
  }

  private take(ch: string) {
    if (this.peek() !== ch) throw new Bail();
    this.pos++;
  }

  private number(): number {
    const m = /^n?\d+/.exec(this.s.slice(this.pos));
    if (!m) throw new Bail();
    this.pos += m[0].length;
    return Number(m[0].replace("n", "-"));
  }

  private seqId() {
    while (/[0-9A-Z]/.test(this.peek())) this.pos++;
    this.take("_");
  }

  private sourceName(): string {
    const len = this.number();
    if (len <= 0 || this.pos + len > this.s.length) throw new Bail();
    const id = this.s.slice(this.pos, this.pos + len);
    this.pos += len;
    return id.startsWith("_GLOBAL__N") ? ANON : id;
  }

  encoding() {
    this.take("_");
    this.take("Z");
    this.name(this.parts);
  }

  /** Reads a <name> into `into`, dropping template arguments; captures the first argument of the last component. */
  private name(into: string[]) {
    const c = this.peek();
    if (c === "N") {
      this.nested(into);
    } else if (c === "Z") {
      this.local(into);
    } else {
      if (c === "S" && this.peek(1) === "t") {
        this.pos += 2;
        into.push("std");
      }
      this.unqualified(into);
      if (this.peek() === "I") this.templateArgs(into === this.parts);
    }
  }

  private nested(into: string[]) {
    this.take("N");
    while ("rVKRO".includes(this.peek()) && this.peek() !== "") this.pos++;
    while (this.peek() !== "E") {
      const c = this.peek();
      if (c === "") throw new Bail();
      if (c === "I") {
        this.templateArgs(into === this.parts);
      } else if (c === "S") {
        if (this.peek(1) === "t") {
          this.pos += 2;
          into.push("std");
        } else if ("absiod".includes(this.peek(1))) {
          this.pos += 2;
          into.push("std");
        } else {
          this.pos++;
          this.seqId();
        }
      } else if (c === "T") {
        this.pos++;
        this.seqId();
      } else if (c === "M") {
        this.pos++;
      } else {
        this.unqualified(into);
      }
    }
    this.pos++;
  }

  private unqualified(into: string[]) {
    const c = this.peek();
    if (c === "L") this.pos++;
    const d = this.peek();
    if (/\d/.test(d)) {
      into.push(this.sourceName());
    } else if (d === "C" || d === "D") {
      if (d === "D" && !/[0-5]/.test(this.peek(1))) throw new Bail();
      const cls = into.at(-1) ?? "";
      into.push(d === "D" ? `~${cls}` : cls);
      this.pos += 2;
    } else if (d === "U") {
      this.pos++;
      if (this.peek() === "t") {
        this.pos++;
        if (this.peek() !== "_") this.number();
        this.take("_");
        into.push("{unnamed type}");
      } else if (this.peek() === "l") {
        this.pos++;
        while (this.peek() !== "E") this.type();
        this.pos++;
        if (this.peek() !== "_") this.number();
        this.take("_");
        into.push("{lambda}");
      } else {
        throw new Bail();
      }
    } else if (into.length > 0 && /[a-z]/.test(d) && /[a-z]/.test(this.peek(1))) {
      const op = this.s.slice(this.pos, this.pos + 2);
      this.pos += 2;
      into.push(op === "cl" ? "operator()" : `operator ${op}`);
    } else {
      throw new Bail();
    }
    while (this.peek() === "B") {
      this.pos++;
      this.sourceName();
    }
  }

  private local(into: string[]) {
    this.take("Z");
    const fn: string[] = [];
    this.name(fn);
    while (this.peek() !== "E") this.type();
    this.pos++;
    into.push(...fn);
    if (this.peek() === "s") this.pos++;
    else this.name(into);
    if (this.peek() === "_") {
      this.pos++;
      if (this.peek() === "_") {
        this.pos++;
        this.number();
        this.take("_");
      } else {
        this.number();
      }
    }
  }

  private templateArgs(capture: boolean) {
    this.take("I");
    let first = true;
    while (this.peek() !== "E") {
      if (this.peek() === "") throw new Bail();
      const start = this.pos;
      this.templateArg();
      if (capture && first) this.firstArg = /^\d/.test(this.s.slice(start)) ? new Itanium(this.s.slice(start, this.pos)).leadingSourceName() : null;
      first = false;
    }
    this.pos++;
  }

  leadingSourceName(): string | null {
    try {
      return this.sourceName();
    } catch {
      return null;
    }
  }

  private templateArg() {
    const c = this.peek();
    if (c === "L") {
      this.exprPrimary();
    } else if (c === "J") {
      this.pos++;
      while (this.peek() !== "E") this.templateArg();
      this.pos++;
    } else if (c === "X") {
      throw new Bail();
    } else {
      this.type();
    }
  }

  private exprPrimary() {
    this.take("L");
    if (this.peek() === "_" && this.peek(1) === "Z") {
      this.pos += 2;
      this.name([]);
      while (this.peek() !== "E") this.type();
    } else {
      this.type();
      while (/[0-9a-fn_.]/.test(this.peek()) && this.peek() !== "") this.pos++;
    }
    this.take("E");
  }

  private type() {
    const c = this.peek();
    if ("vwbcahstijlmxynofdegz".includes(c) && c !== "") {
      this.pos++;
      return;
    }
    switch (c) {
      case "P":
      case "R":
      case "O":
      case "K":
      case "V":
      case "r":
      case "C":
      case "G":
        this.pos++;
        this.type();
        return;
      case "U":
        this.pos++;
        this.sourceName();
        if (this.peek() === "I") this.templateArgs(false);
        this.type();
        return;
      case "D":
        this.builtinD();
        return;
      case "N":
        this.nested([]);
        return;
      case "Z":
        this.local([]);
        return;
      case "S":
        this.pos++;
        if (this.peek() === "t") {
          this.pos++;
          this.unqualified([]);
        } else if ("absiod".includes(this.peek()) && this.peek() !== "") {
          this.pos++;
        } else {
          this.seqId();
        }
        if (this.peek() === "I") this.templateArgs(false);
        return;
      case "T":
        this.pos++;
        this.seqId();
        if (this.peek() === "I") this.templateArgs(false);
        return;
      case "F":
        this.pos++;
        if (this.peek() === "Y") this.pos++;
        while (this.peek() !== "E") {
          if ((this.peek() === "R" || this.peek() === "O") && this.peek(1) === "E") this.pos++;
          else this.type();
        }
        this.pos++;
        return;
      case "A":
        this.pos++;
        if (this.peek() !== "_") this.number();
        this.take("_");
        this.type();
        return;
      case "M":
        this.pos++;
        this.type();
        this.type();
        return;
      default:
        if (/\d/.test(c)) {
          this.sourceName();
          if (this.peek() === "I") this.templateArgs(false);
          return;
        }
        throw new Bail();
    }
  }

  private builtinD() {
    this.pos++;
    const c = this.peek();
    this.pos++;
    if ("dfehiusacn".includes(c) && c !== "") return;
    if (c === "p") {
      this.type();
      return;
    }
    if (c === "F") {
      this.number();
      this.take("_");
      return;
    }
    if (c === "v") {
      this.number();
      this.take("_");
      this.type();
      return;
    }
    throw new Bail();
  }
}

function fromParts(parts: readonly string[], full: string, firstArg: string | null = null): KernelName {
  const named = parts.filter((p) => p !== ANON);
  const base = named.at(-1);
  if (base === undefined) return { scope: "", base: full, full };
  const scope = named.slice(0, -1).map((p) => `${p}::`).join("");
  return WRAPPERS.has(`${scope}${base}`) && firstArg !== null ? { scope: "", base: firstArg, full } : { scope, base, full };
}

function demangle(full: string): KernelName {
  const p = new Itanium(full);
  try {
    p.encoding();
  } catch (e) {
    if (!(e instanceof Bail)) throw e;
  }
  return p.parts.length > 0 ? fromParts(p.parts, full, p.firstArg) : { scope: "", base: full, full };
}

/** Already-demangled text such as `void ns::(anonymous namespace)::k<4, float>(float*, int)`: drop the return type, template arguments, and parameters. */
function fromText(full: string): KernelName {
  const text = full.replaceAll(ANON, "\u0000");
  let depth = 0;
  let out = "";
  for (const ch of text) {
    if (ch === "<") depth++;
    else if (ch === ">") depth = Math.max(0, depth - 1);
    else if (depth === 0) {
      if (ch === "(") break;
      out += ch;
    }
  }
  const name = out.trim().split(" ").at(-1) ?? "";
  const parts = name
    .split("::")
    .filter(Boolean)
    .map((p) => (p === "\u0000" ? ANON : p));
  return fromParts(parts, full);
}

export function simplifyKernelName(name: string): KernelName {
  const full = name.trim();
  if (full.startsWith("_Z")) return demangle(full);
  if (full.includes("::") || full.includes("(")) return fromText(full);
  return { scope: "", base: full, full };
}

export const kernelDisplayName = (name: string) => {
  const k = simplifyKernelName(name);
  return `${k.scope}${k.base}`;
};
