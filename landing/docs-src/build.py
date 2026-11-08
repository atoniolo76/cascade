"""Rebuild the docs site with `uv run python landing/docs-src/build.py` from the repo root."""

# ruff: noqa: E501
from __future__ import annotations

import ast
import html
import io
import json
import re
import shutil
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

from cascade import models

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs"
EXAMPLES = HERE / "examples"
sys.path.insert(0, str(HERE))

from endpoints import ENDPOINTS, UNAUTHORIZED, Endpoint, Param  # noqa: E402
from html_to_md import to_markdown  # noqa: E402
from typenotes import (  # noqa: E402
    ENUM_NOTES,
    FIELD_NOTES,
    MODEL_GROUPS,
    RUN_STATE_CASES,
    SETTINGS_NOTES,
)

NAV: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Getting started", ("index", "quickstart", "agents", "concepts")),
    (
        "Guides",
        ("deploy", "trace-a-workload", "traces", "runs", "reading-results", "simulator"),
    ),
    (
        "API reference",
        ("api", "api-traces", "api-runs", "api-events", "api-simulator", "types", "errors"),
    ),
    ("Help", ("troubleshooting", "limits")),
)

GITHUB = "https://github.com/atoniolo76/cascade"
ROOT = HERE.parent.parent
PROMPT_TEMPLATE = HERE / "agent-prompt.md"
STAND_IN_LOG = ROOT / "tests/fixtures/logs/babelstream_sm86_rtx3070.log"
SIMULATOR_PAGE = "api-simulator"


@dataclass(frozen=True)
class Page:
    slug: str
    group: str
    title: str
    nav: str
    description: str
    source: str

    @property
    def href(self) -> str:
        return f"{self.slug}.html"


@dataclass(frozen=True)
class Heading:
    level: int
    id: str
    text: str


def load_pages() -> list[Page]:
    pages = []
    for group, slugs in NAV:
        for slug in slugs:
            raw = (HERE / "pages" / f"{slug}.html").read_text()
            m = re.match(r"<!--\n(.*?)\n-->\n", raw, re.S)
            if not m:
                raise SystemExit(f"pages/{slug}.html has no metadata block")
            meta = dict(line.split(": ", 1) for line in m.group(1).splitlines())
            pages.append(
                Page(
                    slug=slug,
                    group=group,
                    title=meta["title"],
                    nav=meta.get("nav", meta["title"]),
                    description=meta["description"],
                    source=raw[m.end() :],
                )
            )
    return pages


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def code_span(text: str) -> str:
    """Inline code; a bare documented type name links to its entry on the Types page."""
    if text in TYPE_NAMES:
        return f'<a href="types.html#{text.lower()}"><code>{text}</code></a>'
    return f"<code>{esc(text)}</code>"


def inline(text: str) -> str:
    """Backticks become <code> and [text](href) becomes a link; everything else is trusted HTML."""
    text = re.sub(r"`([^`\n]+)`", lambda m: code_span(m.group(1)), text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)


TOKEN_RES = {
    "bash": re.compile(r"(?P<c>(?<!\S)#[^\n]*)|(?P<s>\"(?:\\.|[^\"\\])*\"|'[^']*')"),
    "python": re.compile(
        r"(?P<c>#[^\n]*)|(?P<s>[rf]?\"(?:\\.|[^\"\\\n])*\"|[rf]?'(?:\\.|[^'\\\n])*')"
        r"|(?P<k>\b(?:import|from|as|with|for|in|if|not|def|return|print|while|True|False|None)\b)"
    ),
    "js": re.compile(
        r"(?P<c>//[^\n]*)|(?P<s>\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`[^`]*`)"
        r"|(?P<k>\b(?:const|let|await|async|new|for|of|if|return|true|false|null)\b)"
    ),
}


def highlight(code: str, lang: str) -> str:
    pattern = TOKEN_RES.get(lang)
    if pattern is None:
        return esc(code)
    out, pos = [], 0
    for m in pattern.finditer(code):
        out.append(esc(code[pos : m.start()]))
        out.append(f'<span class="tk-{m.lastgroup}">{esc(m.group())}</span>')
        pos = m.end()
    out.append(esc(code[pos:]))
    return "".join(out)


def code_block(code: str, lang: str = "", label: str | None = None, body: str | None = None) -> str:
    shown = label or lang or "text"
    inner = body if body is not None else highlight(code.rstrip("\n"), lang)
    return (
        f'<div class="code"><div class="code-bar"><span>{esc(shown)}</span>'
        '<button class="copy" type="button">Copy</button></div>'
        f"<pre><code>{inner}</code></pre></div>"
    )


def tabs(panels: list[tuple[str, str]]) -> str:
    buttons = "".join(
        f'<button type="button" role="tab" data-tab="{esc(label)}"'
        f' aria-selected="{"true" if i == 0 else "false"}">{esc(label)}</button>'
        for i, (label, _) in enumerate(panels)
    )
    bodies = "".join(
        f'<div class="tab-panel" role="tabpanel" data-tab="{esc(label)}"{" hidden" if i else ""}>'
        f"{body}</div>"
        for i, (label, body) in enumerate(panels)
    )
    return f'<div class="tabs"><div class="tab-list" role="tablist">{buttons}</div>{bodies}</div>'


def trimmed_note(value: object) -> str | None:
    if isinstance(value, dict) and set(value) == {"$trimmed"}:
        return value["$trimmed"]
    return None


def json_html(value: object, indent: int = 0) -> str:
    pad, inner = "  " * indent, "  " * (indent + 1)

    def comment(text: str) -> str:
        return f'<span class="tk-c">/* {esc(text)} trimmed */</span>'

    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = []
        items = list(value.items())
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 and items[i + 1][0] != "$trimmed" else ""
            if k == "$trimmed":
                rows.append(f"{inner}{comment(v)}")
            else:
                key = f'<span class="tk-k">{esc(json.dumps(k))}</span>'
                rows.append(f"{inner}{key}: {json_html(v, indent + 1)}{comma}")
        return "{\n" + "\n".join(rows) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        rows = []
        for i, v in enumerate(value):
            note = trimmed_note(v)
            nxt = trimmed_note(value[i + 1]) if i < len(value) - 1 else "end"
            comma = "," if nxt is None else ""
            rows.append(
                f"{inner}{comment(note)}" if note else f"{inner}{json_html(v, indent + 1)}{comma}"
            )
        return "[\n" + "\n".join(rows) + f"\n{pad}]"
    if isinstance(value, str):
        return f'<span class="tk-s">{esc(json.dumps(value, ensure_ascii=False))}</span>'
    if value is None or isinstance(value, bool):
        return f'<span class="tk-l">{json.dumps(value)}</span>'
    return f'<span class="tk-n">{json.dumps(value)}</span>'


def json_text(value: object) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(json_html(value)))


STATUS_TEXT = {
    200: "OK",
    201: "Created",
    204: "No Content",
    401: "Unauthorized",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
}


def example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text())


def response_block(name: str, label: str | None = None) -> str:
    ex = example(name)
    status = f"{ex['status']} {STATUS_TEXT[ex['status']]}"
    head = label or "Response"
    meta = f"{head} · {status}" + (f" · {ex['content_type']}" if ex["content_type"] else "")
    if ex["body"] is None:
        return code_block("", label=meta, body='<span class="tk-c">(empty body)</span>')
    if isinstance(ex["body"], str):
        text = re.sub(r"^\$trimmed (.*)$", r"/* \1 trimmed */", ex["body"], flags=re.M)
        body = re.sub(
            r"/\* (.*?) trimmed \*/", lambda m: f'<span class="tk-c">{m.group(0)}</span>', esc(text)
        )
        return code_block(text, label=meta, body=body)
    return code_block(json_text(ex["body"]), label=meta, body=json_html(ex["body"]))


@dataclass(frozen=True)
class FieldInfo:
    name: str
    annotation: str
    comment: str | None


@dataclass(frozen=True)
class ModelInfo:
    name: str
    base: str | None
    doc: str | None
    fields: tuple[FieldInfo, ...]


def _strip_annotated(node: ast.expr) -> ast.expr:
    class Strip(ast.NodeTransformer):
        def visit_Subscript(self, n: ast.Subscript) -> ast.AST:
            self.generic_visit(n)
            if isinstance(n.value, ast.Name) and n.value.id == "Annotated":
                return n.slice.elts[0] if isinstance(n.slice, ast.Tuple) else n.slice
            return n

    return Strip().visit(node)


def read_models() -> tuple[dict[str, ModelInfo], dict[str, str], dict[int, str]]:
    src = Path(models.__file__).read_text()
    comments = {
        tok.start[0]: tok.string.lstrip("#").strip()
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type == tokenize.COMMENT
    }
    infos: dict[str, ModelInfo] = {}
    aliases: dict[str, str] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef):
            fields = tuple(
                FieldInfo(
                    stmt.target.id,
                    ast.unparse(_strip_annotated(stmt.annotation)).replace("'", '"'),
                    comments.get(stmt.end_lineno),
                )
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            )
            base = ast.unparse(node.bases[0]) if node.bases else None
            infos[node.name] = ModelInfo(node.name, base, ast.get_docstring(node), fields)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name[0].isupper() and not name.isupper():
                aliases[name] = ast.unparse(_strip_annotated(node.value)).replace("'", '"')
    return infos, aliases, comments


MODELS, ALIASES, COMMENTS = read_models()
TYPE_NAMES = set(MODELS) | set(ALIASES)


def type_link(name: str) -> str:
    return f'<a href="types.html#{name.lower()}">{name}</a>'


def type_html(annotation: str) -> str:
    return re.sub(
        r"\b[A-Z]\w*\b",
        lambda m: type_link(m.group()) if m.group() in TYPE_NAMES else m.group(),
        esc(annotation),
    )


def constraints(model: type[BaseModel], field: str, defaults: bool = True) -> list[str]:
    info = model.model_fields[field]
    out = []
    words = {
        "ge": "at least {}",
        "le": "at most {}",
        "gt": "more than {}",
        "lt": "less than {}",
        "min_length": "at least {} item(s)",
    }
    for meta in info.metadata:
        for attr, fmt in words.items():
            if getattr(meta, attr, None) is not None:
                out.append(fmt.format(getattr(meta, attr)).replace("item(s)", "item"))
    if defaults and not info.is_required():
        out.append(f"default `{json.dumps(info.default)}`")
    return out


def sentence_text(comment: str) -> str:
    """A source comment as a sentence: `e.g.` spelled out, capitalized, ending in a period."""
    text = re.sub(r"^e\.g\. ", "For example, ", comment)
    if not re.match(r"\S*[_./]", text):
        text = text[0].upper() + text[1:]
    return text if text.endswith((".", ")")) else text + "."


def sentence(comment: str) -> str:
    return esc(sentence_text(comment))


def fields_table(name: str) -> str:
    model = getattr(models, name)
    rows = []
    for f in MODELS[name].fields:
        note = FIELD_NOTES.get(f"{name}.{f.name}")
        parts = [inline(note) if note else (sentence(f.comment) if f.comment else "")]
        extra = constraints(model, f.name)
        if extra:
            parts.append(f'<span class="constraint">{inline(", ".join(extra))}</span>')
        desc = " ".join(p for p in parts if p)
        rows.append(
            f'<tr><td><code>{f.name}</code></td><td><code class="type">'
            f"{type_html(f.annotation)}</code></td><td>{desc}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table class="table fields"><thead><tr><th>field</th>'
        "<th>type</th><th>description</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def model_section(name: str) -> str:
    info = MODELS[name]
    model = getattr(models, name)
    parts = [f'<h3 id="{name.lower()}"><code>{name}</code></h3>']
    if info.doc:
        parts.append(f"<p>{inline(info.doc)}</p>")
    lead = FIELD_NOTES.get(name)
    if lead:
        parts.append(f"<p>{inline(lead)}</p>")
    if issubclass(model, BaseModel):
        if info.base and info.base in MODELS:
            parts.append(f"<p>Every field of {type_link(info.base)}, plus these.</p>")
        parts.append(fields_table(name))
    else:
        parts.append(enum_table(name))
    return "\n".join(parts)


def enum_table(name: str) -> str:
    enum = getattr(models, name)
    rows = "".join(
        f"<tr><td><code>{m.value}</code></td><td>{inline(ENUM_NOTES[name][m.value])}</td></tr>"
        for m in enum
    )
    return (
        '<div class="table-wrap"><table class="table"><thead><tr><th>value</th><th>meaning</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def alias_section(name: str) -> str:
    parts = [
        f'<h3 id="{name.lower()}"><code>{name}</code></h3>',
        code_block(f"{name} = {ALIASES[name]}", "python", label="cascade/models.py"),
    ]
    if name in ENUM_NOTES:
        values = get_args(getattr(models, name))
        rows = "".join(
            f"<tr><td><code>{v}</code></td><td>{inline(ENUM_NOTES[name][v])}</td></tr>"
            for v in values
        )
        parts.append(
            '<div class="table-wrap"><table class="table"><thead><tr><th>value</th>'
            f"<th>meaning</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    if name in FIELD_NOTES:
        parts.append(f"<p>{inline(FIELD_NOTES[name])}</p>")
    return "\n".join(parts)


def check_type_notes() -> None:
    grouped = [n for _, names in MODEL_GROUPS for n in names]
    documented = set(MODELS) | set(ALIASES)
    missing = documented - set(grouped)
    unknown = set(grouped) - documented
    if missing or unknown or len(grouped) != len(set(grouped)):
        raise SystemExit(f"typenotes.MODEL_GROUPS: missing {missing}, unknown {unknown}")
    for key in FIELD_NOTES:
        model, _, field = key.partition(".")
        fields = {f.name: f for f in MODELS[model].fields} if model in MODELS else {}
        if model not in documented or (field and field not in fields):
            raise SystemExit(f"typenotes.FIELD_NOTES has a stale key {key!r}")
        if field and fields[field].comment:
            raise SystemExit(f"{key} already has a comment in cascade/models.py; drop the note")
    bare = [
        f"{m}.{f.name}"
        for m, info in MODELS.items()
        for f in info.fields
        if not f.comment and f"{m}.{f.name}" not in FIELD_NOTES
    ]
    if bare:
        raise SystemExit(f"add a comment in cascade/models.py or a typenotes note for {bare}")
    for name in (models.TraceState.__name__, models.SimState.__name__):
        values = {m.value for m in getattr(models, name)}
        if set(ENUM_NOTES[name]) != values:
            raise SystemExit(f"typenotes.ENUM_NOTES[{name!r}] must cover exactly {values}")
    if set(ENUM_NOTES["RunState"]) != set(get_args(models.RunState)):
        raise SystemExit("typenotes.ENUM_NOTES['RunState'] must cover every RunState value")


def types_reference() -> str:
    out = []
    for group, names in MODEL_GROUPS:
        out.append(f"<h2>{esc(group)}</h2>")
        for name in names:
            out.append(alias_section(name) if name in ALIASES else model_section(name))
    return "\n".join(out)


S = models.SimState
TERMINAL_EXITS = (S.failed, S.cancelled)
SIM_POS = {
    S.pending: (80, 60),
    S.launching: (250, 60),
    S.running: (440, 60),
    S.succeeded: (640, 60),
    S.lost: (345, 175),
}
SIM_BEND = {(S.launching, S.succeeded): -62, (S.running, S.lost): 26, (S.lost, S.running): 26}
TRACE_POS = {
    "uploading": (90, 60),
    "processing": (300, 60),
    "ready": (510, 60),
    "failed": (300, 165),
}
# From cascade/ingest.py: Ingestor._advance_upload and Ingestor._process.
TRACE_EDGES = (
    ("uploading", "processing"),
    ("processing", "ready"),
    ("uploading", "failed"),
    ("processing", "failed"),
)
NODE_W, NODE_H = 112, 36


def _exit(center: tuple[float, float], toward: tuple[float, float]) -> tuple[float, float]:
    (cx, cy), (tx, ty) = center, toward
    dx, dy = tx - cx, ty - cy
    scale = min(
        (NODE_W / 2 + 4) / abs(dx) if dx else float("inf"),
        (NODE_H / 2 + 4) / abs(dy) if dy else float("inf"),
    )
    return cx + dx * scale, cy + dy * scale


def state_svg(
    title: str,
    pos: dict,
    edges: list[tuple[object, object]],
    bends: dict,
    height: int,
    tone: dict[str, str],
    footer: list[str] = (),
) -> str:
    paths = []
    for a, b in edges:
        (ax, ay), (bx, by) = pos[a], pos[b]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        bend = bends.get((a, b), 14 if (b, a) in edges else 0)
        nx, ny = -(by - ay) / length, (bx - ax) / length
        cx, cy = mx + nx * bend, my + ny * bend
        sx, sy = _exit(pos[a], (cx, cy))
        ex, ey = _exit(pos[b], (cx, cy))
        paths.append(
            f'<path d="M{sx:.1f} {sy:.1f} Q{cx:.1f} {cy:.1f} {ex:.1f} {ey:.1f}" '
            'marker-end="url(#arrow)"/>'
        )
    nodes = []
    for state, (x, y) in pos.items():
        cls = tone.get(str(state), "")
        nodes.append(
            f'<g class="dg-node {cls}"><rect x="{x - NODE_W / 2}" y="{y - NODE_H / 2}" '
            f'width="{NODE_W}" height="{NODE_H}" rx="10"/>'
            f'<text x="{x}" y="{y + 4}" text-anchor="middle">{state}</text></g>'
        )
    foot = "".join(
        f'<text class="dg-note" x="20" y="{height - 14 - 20 * (len(footer) - 1 - i)}">{esc(t)}</text>'
        for i, t in enumerate(footer)
    )
    return (
        f'<figure class="diagram"><svg viewBox="0 0 720 {height}" role="img" '
        f'aria-label="{esc(title)}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" '
        'refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10z" class="dg-head"/></marker></defs>'
        f'<g class="dg-edges">{"".join(paths)}</g>{"".join(nodes)}{foot}</svg>'
        f"<figcaption>{esc(title)}</figcaption></figure>"
    )


def sim_diagram() -> str:
    edges = [
        (a, b)
        for a, targets in models.SIM_TRANSITIONS.items()
        for b in sorted(targets, key=list(S).index)
        if b not in TERMINAL_EXITS
    ]
    missing = {s for e in edges for s in e} - set(SIM_POS)
    if missing:
        raise SystemExit(f"SIM_POS has no position for {missing}")
    footer = []
    for target in TERMINAL_EXITS:
        sources = [a.value for a, t in models.SIM_TRANSITIONS.items() if target in t]
        footer.append(f"{target.value} ← {', '.join(sources)}")
    return state_svg(
        "Sim states, drawn from SIM_TRANSITIONS in cascade/models.py",
        SIM_POS,
        edges,
        SIM_BEND,
        290,
        {"succeeded": "good", "lost": "warn"},
        footer,
    )


def trace_diagram() -> str:
    return state_svg(
        "Trace states, from cascade/ingest.py",
        TRACE_POS,
        list(TRACE_EDGES),
        {},
        210,
        {"ready": "good", "failed": "bad"},
    )


def transitions_table() -> str:
    states = list(S)
    head = "".join(f"<th><code>{s.value}</code></th>" for s in states)
    rows = []
    for a in states:
        cells = "".join(
            '<td class="yes">✓</td>' if b in models.SIM_TRANSITIONS[a] else '<td class="no">·</td>'
            for b in states
        )
        rows.append(f"<tr><td><code>{a.value}</code></td>{cells}</tr>")
    return (
        '<div class="table-wrap"><table class="table matrix"><thead><tr><th>from ↓ to →</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def transitions_owner_comment() -> str:
    for text in COMMENTS.values():
        if text.startswith("Allowed sim transitions"):
            return text
    raise SystemExit("the SIM_TRANSITIONS comment in cascade/models.py moved")


def run_state_table() -> str:
    rows = []
    for states in RUN_STATE_CASES:
        result = models.run_state([S(s) for s in states])
        rows.append(
            f"<tr><td>{', '.join(f'<code>{s}</code>' for s in states)}</td>"
            f"<td><code>{result}</code></td></tr>"
        )
    return (
        '<div class="table-wrap"><table class="table"><thead><tr><th>sim states</th>'
        f"<th>run state</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def path_html(path: str) -> str:
    return re.sub(r"\{(\w+)\}", r'<span class="pp">{\1}</span>', esc(path))


def params_table(params: tuple[Param, ...]) -> str:
    rows = "".join(
        f"<tr><td><code>{p.name}</code>{'' if p.required else ' <span class="opt">optional</span>'}"
        f'</td><td><code class="type">{type_html(p.type)}</code></td><td>{inline(p.doc)}</td></tr>'
        for p in params
    )
    return (
        '<div class="table-wrap"><table class="table fields"><thead><tr><th>name</th>'
        f"<th>type</th><th>description</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def samples(ep: Endpoint) -> str:
    ex = example(ep.example)
    method = ep.method
    if ep.kind == "sse":
        path = ep.path.replace("{run_id}", ex["run_id"])
        return tabs(
            [
                (
                    "curl",
                    code_block(
                        f'curl -N -H "Authorization: Bearer $CASCADE_TOKEN" \\\n  "$CASCADE_URL{path}"',
                        "bash",
                    ),
                ),
                ("Python", code_block(SSE_PY.format(path=path), "python")),
                ("JavaScript", code_block(SSE_JS.format(path=path), "js")),
            ]
        )
    path = ex["request"]["path"]
    body = ex["request"]["body"]
    token = "$CASCADE_TOKEN" if ep.auth != "none" else None
    curl = ["curl"]
    if method != "GET":
        curl.append(f"-X {method}")
    if ep.kind == "download":
        curl.append("-OJ")
    curl_lines = [" ".join(curl) + f' "$CASCADE_URL{path}"']
    if token:
        curl_lines.append(f'-H "Authorization: Bearer {token}"')
    if body is not None:
        curl_lines.append('-H "Content-Type: application/json"')
        curl_lines.append(f"-d '{json.dumps(body)}'")
    curl_code = " \\\n  ".join(curl_lines)

    py = ["import os", "", "import httpx", ""]
    if token:
        py.append('headers = {"Authorization": f"Bearer {os.environ[\'CASCADE_TOKEN\']}"}')
    call_args = [f"f\"{{os.environ['CASCADE_URL']}}{path}\""]
    if token:
        call_args.append("headers=headers")
    if body is not None:
        call_args.append(f"json={py_literal(body)}")
    py.append(f"resp = httpx.{method.lower()}({', '.join(call_args)})")
    py.append("resp.raise_for_status()")
    if ep.kind == "download" and ex["content_type"] == "text/plain":
        py.append(f'open("{path.split("/")[3]}.log", "w").write(resp.text)')
    elif ep.kind == "download":
        py.append(f'open("{path.split("/")[3]}-results.json", "wb").write(resp.content)')
    elif ex["status"] != 204:
        py.append("print(resp.json())")

    opts = [f'method: "{method}"'] if method != "GET" else []
    headers = []
    if token:
        headers.append("Authorization: `Bearer ${CASCADE_TOKEN}`")
    if body is not None:
        headers.append('"Content-Type": "application/json"')
    if headers:
        opts.append("headers: { " + ", ".join(headers) + " }")
    if body is not None:
        opts.append(f"body: JSON.stringify({json.dumps(body)})")
    js = [f"const resp = await fetch(`${{CASCADE_URL}}{path}`"]
    if opts:
        js[0] += ", {"
        js += [f"  {o}," for o in opts]
        js.append("});")
    else:
        js[0] += ");"
    if ep.kind == "download" and ex["content_type"] == "text/plain":
        js.append("const log = await resp.text();")
    elif ep.kind == "download":
        js.append("const results = await resp.json();")
    elif ex["status"] != 204:
        js.append("console.log(await resp.json());")
    return tabs(
        [
            ("curl", code_block(curl_code, "bash")),
            ("Python", code_block("\n".join(py), "python")),
            ("JavaScript", code_block("\n".join(js), "js")),
        ]
    )


def py_literal(value: object) -> str:
    return (
        json.dumps(value).replace("true", "True").replace("false", "False").replace("null", "None")
    )


SSE_PY = """import json
import os

import httpx

headers = {{"Authorization": f"Bearer {{os.environ['CASCADE_TOKEN']}}"}}
url = f"{{os.environ['CASCADE_URL']}}{path}"
with httpx.stream("GET", url, headers=headers, timeout=None) as resp:
    for line in resp.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            print(event["type"])"""

SSE_JS = """// EventSource cannot send headers, so pass the token as a query parameter.
const url = `${{CASCADE_URL}}{path}?token=${{encodeURIComponent(CASCADE_TOKEN)}}`;
const events = new EventSource(url);
events.onmessage = (msg) => {{
  const event = JSON.parse(msg.data);
  console.log(event.type);
}};"""


def endpoint_section(ep: Endpoint) -> str:
    parts = [
        f'<section class="endpoint"><h2 id="{ep.key}">{esc(ep.title)}</h2>',
        f'<div class="route"><span class="method {ep.method.lower()}">{ep.method}</span>'
        f'<span class="path">{path_html(ep.path)}</span>'
        f'<span class="auth">{AUTH_LABEL[ep.auth]}</span></div>',
        f"<p>{inline(ep.doc)}</p>",
    ]
    for label, where in (("Path parameters", "path"), ("Query parameters", "query")):
        ps = tuple(p for p in ep.params if p.where == where)
        if ps:
            parts.append(f"<h3 data-notoc>{label}</h3>{params_table(ps)}")
    if ep.body:
        parts.append(f"<h3 data-notoc>Request body</h3><p>JSON, {type_link(ep.body)}.</p>")
        parts.append(fields_table(ep.body))
        for nested in ep.nested:
            parts.append(f"<p>Each {type_link(nested)} has these fields.</p>{fields_table(nested)}")
    parts.append(f"<h3 data-notoc>Returns</h3><p>{inline(ep.returns)}</p>")
    if ep.errors:
        rows = "".join(
            f'<tr><td><a href="errors.html#e{code}"><code>{code}</code></a></td><td>{inline(when)}</td></tr>'
            for code, when in ep.errors
        )
        parts.append(
            '<h3 data-notoc>Errors</h3><div class="table-wrap"><table class="table"><thead><tr>'
            f"<th>status</th><th>when</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    parts.append("<h3 data-notoc>Example</h3>")
    parts.append(samples(ep))
    if ep.kind == "sse":
        parts.append(ep.extra or "")
    else:
        parts.append(response_block(ep.example))
        for label, name in ep.more_examples:
            parts.append(response_block(name, label))
    parts.append("</section>")
    return "\n".join(parts)


AUTH_LABEL = {
    "bearer": "bearer token",
    "none": "no auth",
    "bearer-or-query": "bearer token or ?token=",
}


def check_routes() -> None:
    """Fails the build when the app serves an /api route that no Endpoint documents."""
    from cascade.api import create_app
    from cascade.settings import Settings

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings.from_env({"CASCADE_TOKEN": "x", "CASCADE_DATA_DIR": tmp})
        app = create_app(settings)
        served = {
            (m, r.path)
            for r in app.routes
            if getattr(r, "path", "").startswith("/api/") and "{path:path}" not in r.path
            for m in getattr(r, "methods", ())
        }
        app.state.cascade.store.close()
    documented = {(e.method, e.path) for e in ENDPOINTS}
    if served != documented:
        raise SystemExit(
            f"routes not documented: {sorted(served - documented)}; "
            f"documented but not served: {sorted(documented - served)}"
        )


def endpoints_for(page: str) -> str:
    return "\n".join(endpoint_section(e) for e in ENDPOINTS if e.page == page)


def endpoint_index() -> str:
    rows = "".join(
        f'<tr><td><span class="method {e.method.lower()}">{e.method}</span></td>'
        f'<td><a href="{e.page}.html#{e.key}"><code>{esc(e.path)}</code></a></td>'
        f"<td>{inline(e.summary)}</td></tr>"
        for e in ENDPOINTS
    )
    return (
        '<div class="table-wrap"><table class="table routes"><thead><tr><th>method</th>'
        f"<th>path</th><th>what it does</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def env_vars() -> dict[str, str | None]:
    """Every `env.get("CASCADE_...", default)` in cascade/settings.py, with its default."""
    from cascade import settings

    found: dict[str, str | None] = {}
    for node in ast.walk(ast.parse(Path(settings.__file__).read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "env"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            default = node.args[1] if len(node.args) > 1 else None
            found[node.args[0].value] = default.value if isinstance(default, ast.Constant) else None
    return found


def settings_table(arg: str | None = None) -> str:
    found = env_vars()
    if set(found) != set(SETTINGS_NOTES):
        raise SystemExit(
            f"typenotes.SETTINGS_NOTES must cover exactly the env vars in cascade/settings.py: "
            f"missing {set(found) - set(SETTINGS_NOTES)}, stale {set(SETTINGS_NOTES) - set(found)}"
        )
    rows = []
    for name, default in found.items():
        shown, doc = SETTINGS_NOTES[name]
        if shown is None:
            shown = f"<code>{esc(default)}</code>" if default else "none"
        rows.append(
            f"<tr><td><code>{name}</code></td><td>{inline(shown)}</td><td>{inline(doc)}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table class="table"><thead><tr><th>variable</th>'
        f"<th>default</th><th>what it sets</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def event_block(kind: str) -> str:
    events = example("events")
    labels = {"sim_finished": "sim event, sim finished"}
    label = f"data: {labels.get(kind, kind + ' event')}"
    return code_block(
        json_text(events["events"][kind]), label=label, body=json_html(events["events"][kind])
    )


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()).strip("-")


ROUTE_BY_KEY = {e.key: f"`{e.method} {e.path}`" for e in ENDPOINTS}


def plain(text: str) -> str:
    """Docs prose for Markdown read outside the site: a link to a route names the route, any
    other link becomes its text."""
    return re.sub(
        r"\[([^\]\n]+)\]\([^)\s#]*#?([^)\s]*)\)",
        lambda m: ROUTE_BY_KEY.get(m.group(2), m.group(1)),
        text,
    )


def or_list(items: list[str]) -> str:
    return ", ".join(items[:-1]) + f" or {items[-1]}" if len(items) > 1 else items[0]


def agent_endpoints() -> list[Endpoint]:
    return [e for e in ENDPOINTS if e.page != SIMULATOR_PAGE]


def prompt_route_index() -> str:
    rows = [f"| {e.method} | `{e.path}` | {plain(e.summary)} |" for e in agent_endpoints()]
    return "\n".join(["| method | path | what it does |", "|---|---|---|", *rows])


PROMPT_AUTH = {
    "bearer": None,
    "none": "No token needed.",
    "bearer-or-query": "Bearer token, or the token as the `token` query parameter.",
}


def prompt_route(ep: Endpoint) -> str:
    lines = [f"### {ep.method} {ep.path}", "", f"{ep.title}. {plain(ep.doc)}", ""]
    if PROMPT_AUTH[ep.auth]:
        lines.append(f"- Auth: {PROMPT_AUTH[ep.auth]}")
    for p in ep.params:
        opt = ", optional" if not p.required else ""
        lines.append(
            f"- {p.where.capitalize()} parameter `{p.name}` ({p.type}{opt}): {plain(p.doc)}"
        )
    if ep.body:
        lines.append(f"- Body: `{ep.body}`.")
    lines.append(f"- Returns: {plain(ep.returns)}")
    for code, when in ep.errors:
        if (code, when) != UNAUTHORIZED:
            lines.append(f"- `{code}`: {plain(when)}")
    return "\n".join(lines)


def prompt_routes() -> str:
    return "\n\n".join(prompt_route(e) for e in agent_endpoints())


def prompt_simulator_routes() -> str:
    routes = or_list([f"`{e.method} {e.path}`" for e in ENDPOINTS if e.page == SIMULATOR_PAGE])
    return (
        f"- Never call {routes}. Those routes are for simulators, and a heartbeat changes a sim's "
        "state."
    )


def prompt_states() -> str:
    def items(enum: str, values: list[str], nexts: dict[str, list[str]]) -> list[str]:
        out = []
        for v in values:
            note = plain(ENUM_NOTES[enum][v])
            follow = nexts.get(v)
            if follow:
                note += f" Next: {or_list([f'`{n}`' for n in follow])}."
            elif not note.endswith("Terminal."):
                note += " Terminal."
            out.append(f"- `{v}`: {note}")
        return out

    trace_next: dict[str, list[str]] = {}
    for a, b in TRACE_EDGES:
        trace_next.setdefault(a, []).append(b)
    sim_next = {
        a.value: [b.value for b in S if b in targets]
        for a, targets in models.SIM_TRANSITIONS.items()
    }
    trace_values = [m.value for m in models.TraceState]
    run_values = list(get_args(models.RunState))
    return "\n".join(
        [
            "### Trace states",
            "",
            "A trace's `state`. Runs can only launch on a `ready` trace.",
            "",
            *items("TraceState", trace_values, trace_next),
            "",
            "### Sim states",
            "",
            "A sim's `state`. The app ignores any move not listed here.",
            "",
            *items("SimState", [s.value for s in S], sim_next),
            "",
            "### Run states",
            "",
            plain(FIELD_NOTES["RunState"]),
            "",
            *(f"- `{v}`: {plain(ENUM_NOTES['RunState'][v])}" for v in run_values),
        ]
    )


STATE_TYPES = {"TraceState", "SimState", "RunState"}


def type_names_in(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z]\w*\b", text)) & TYPE_NAMES


def prompt_type_names() -> list[str]:
    todo = set()
    for e in agent_endpoints():
        todo |= type_names_in(e.returns) | type_names_in(e.body or "") | set(e.nested)
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in ALIASES:
            todo |= type_names_in(ALIASES[name])
        else:
            info = MODELS[name]
            todo |= type_names_in(info.base or "")
            for f in info.fields:
                todo |= type_names_in(f.annotation)
    order = [n for _, names in MODEL_GROUPS for n in names]
    return [n for n in order if n in seen - STATE_TYPES]


def field_text(model: str, f: FieldInfo) -> str:
    note = FIELD_NOTES.get(f"{model}.{f.name}")
    desc = plain(note) if note else (sentence_text(f.comment) if f.comment else "")
    extra = constraints(getattr(models, model), f.name, defaults=False)
    if extra:
        desc += f" Allowed: {', '.join(extra)}."
    return f"- `{f.name}` (`{f.annotation}`): {desc}".rstrip()


def prompt_type(name: str) -> str:
    if name in ALIASES:
        lead = f"`{name}` is `{ALIASES[name]}`."
        note = FIELD_NOTES.get(name)
        return f"### {name}\n\n{lead} {plain(note) if note else ''}".rstrip()
    info = MODELS[name]
    lead = [plain(t) for t in (info.doc, FIELD_NOTES.get(name)) if t]
    if info.base in MODELS:
        lead.append(f"Every field of `{info.base}`, plus these.")
    body = "\n".join(field_text(name, f) for f in info.fields)
    return "\n\n".join([f"### {name}", *([" ".join(lead)] if lead else []), body])


def prompt_types() -> str:
    return "\n\n".join(prompt_type(n) for n in prompt_type_names())


def stand_in_kernels() -> str:
    names: list[str] = []
    for m in re.finditer(r"^kernel_name = _Z(\d+)(\w+)", STAND_IN_LOG.read_text(), re.M):
        ident = m.group(2)[: int(m.group(1))]
        if ident not in names:
            names.append(ident)
    if not names:
        raise SystemExit(f"no kernel names in {STAND_IN_LOG}")
    quoted = [f"`{n}`" for n in names]
    return ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


PROMPT_PARTS = {
    "route-index": prompt_route_index,
    "routes": prompt_routes,
    "simulator-routes": prompt_simulator_routes,
    "states": prompt_states,
    "types": prompt_types,
    "stand-in-kernels": stand_in_kernels,
}


def check_prompt_template(template: str) -> None:
    """The hand-written parts of the prompt may only name routes the app serves and real fields."""
    served = {(e.method, e.path) for e in ENDPOINTS}
    for method, path in re.findall(r"\b(GET|POST) (/api/[^\s`]*)", template):
        if (method, path) not in served:
            raise SystemExit(f"agent-prompt.md names {method} {path}, which no Endpoint documents")
    fields = {f.name for info in MODELS.values() for f in info.fields}
    kernels = set(re.findall(r"`(\w+)`", stand_in_kernels()))
    for token in re.findall(r"`([^`\n]+)`", template):
        for part in token.split("."):
            if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)+", part) and part not in fields | kernels:
                raise SystemExit(f"agent-prompt.md names `{part}`, which is no model field")
    unknown = set(re.findall(r"\{\{([\w-]+)\}\}", template)) - set(PROMPT_PARTS)
    if unknown:
        raise SystemExit(f"agent-prompt.md has unknown placeholders {unknown}")


@cache
def agent_prompt() -> str:
    template = PROMPT_TEMPLATE.read_text()
    check_prompt_template(template)
    return re.sub(r"\{\{([\w-]+)\}\}", lambda m: PROMPT_PARTS[m.group(1)](), template)


SKILL_DESCRIPTION = (
    "Drive a Cascade app over its HTTP API. Use when the user asks about Cascade traces, GPU "
    "configs, runs or sims, or wants to launch, watch, compare or explain Accel-Sim simulation "
    "results."
)


def skill_file() -> str:
    return f"---\nname: cascade\ndescription: {SKILL_DESCRIPTION}\n---\n\n{agent_prompt()}"


def prompt_block(arg: str | None = None) -> str:
    return (
        '<div class="prompt-block" id="prompt-code">'
        + code_block(agent_prompt(), label="agent-prompt.md")
        + "</div>"
    )


def llms_txt(pages: list[Page]) -> str:
    intro = next(p for p in pages if p.slug == "index").description
    out = [
        "# Cascade docs",
        "",
        f"> {intro} These docs cover running the app, tracing a workload, launching runs, reading results, and the HTTP API.",
        "",
        "To drive a Cascade app from an AI agent, give the agent [agent-prompt.md](agent-prompt.md). It lists every route an agent may call, the state machines, and how to report results.",
    ]
    for group, slugs in NAV:
        out += ["", f"## {group}", ""]
        out += [f"- [{p.title}]({p.href}): {p.description}" for p in pages if p.slug in slugs]
    out += [
        "",
        "## Agent files",
        "",
        "- [Agent prompt](agent-prompt.md): The setup prompt for an AI agent, as Markdown.",
        "- [Cascade skill](cascade-skill/SKILL.md): The same prompt as an agent skill, with `name` and `description` frontmatter.",
        "- [Full docs](llms-full.txt): Every page above as Markdown, in one file.",
    ]
    return "\n".join(out) + "\n"


def llms_full(pages: list[Page], bodies: dict[str, str]) -> str:
    intro = next(p for p in pages if p.slug == "index").description
    parts = [f"# Cascade docs\n\n> {intro}\n"]
    for p in pages:
        parts.append(f"# {p.title}\n\nPage: {p.href}\n\n{to_markdown(bodies[p.slug], p.href)}")
    return "\n\n".join(parts)


DIRECTIVES = {
    "types": lambda arg: types_reference(),
    "sim-diagram": lambda arg: sim_diagram(),
    "trace-diagram": lambda arg: trace_diagram(),
    "transitions-table": lambda arg: transitions_table(),
    "transitions-comment": lambda arg: (
        f"<blockquote><code>{esc(transitions_owner_comment())}</code></blockquote>"
    ),
    "run-state-table": lambda arg: run_state_table(),
    "endpoints": endpoints_for,
    "endpoint-index": lambda arg: endpoint_index(),
    "example": response_block,
    "fields": fields_table,
    "enum": enum_table,
    "settings": settings_table,
    "event": event_block,
    "agent-prompt": prompt_block,
    "snippet": lambda name: code_block((HERE / "snippets" / name).read_text(), "python", name),
}


def render_body(source: str) -> tuple[str, list[Heading]]:
    held: list[str] = []

    def hold(fragment: str) -> str:
        held.append(fragment)
        return f"\x00{len(held) - 1}\x00"

    def fence(m: re.Match) -> str:
        lang, label, code = m.group(1) or "", m.group(2), m.group(3)
        return hold(code_block(code, lang, label))

    text = re.sub(r'^```(\w*)(?: "([^"]*)")?\n(.*?)^```\n', fence, source, flags=re.M | re.S)
    text = re.sub(
        r"\{\{([\w-]+)(?: ([^}]+))?\}\}",
        lambda m: hold(DIRECTIVES[m.group(1)](m.group(2))),
        text,
    )
    text = inline(text)
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], text)

    headings: list[Heading] = []
    seen: dict[str, int] = {}

    def heading(m: re.Match) -> str:
        level, attrs, inner = int(m.group(1)), m.group(2), m.group(3)
        idm = re.search(r'id="([^"]+)"', attrs)
        hid = idm.group(1) if idm else slugify(inner)
        if hid in seen:
            seen[hid] += 1
            hid = f"{hid}-{seen[hid]}"
        else:
            seen[hid] = 1
        attrs = re.sub(r'\s*id="[^"]+"', "", attrs)
        if "data-notoc" not in attrs:
            headings.append(Heading(level, hid, re.sub(r"<[^>]+>", "", inner)))
        attrs = attrs.replace(" data-notoc", "")
        return (
            f'<h{level} id="{hid}"{attrs}>{inner}'
            f'<a class="anchor" href="#{hid}" aria-label="Link to this section">#</a></h{level}>'
        )

    text = re.sub(r"<h([23])([^>]*)>(.*?)</h\1>", heading, text, flags=re.S)
    return text, headings


MARK = (HERE / "mark.svg").read_text().strip()

ICONS = {
    "search": '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/></svg>',
    "menu": '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 4h12M2 8h12M2 12h12"/></svg>',
    "theme": '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5"/><path d="M8 2.5v11a5.5 5.5 0 0 0 0-11z" class="fill"/></svg>',
}


def sidebar(pages: list[Page], current: Page) -> str:
    by_slug = {p.slug: p for p in pages}
    groups = []
    for group, slugs in NAV:
        links = "".join(
            f'<li><a href="{by_slug[s].href}"'
            f"{' aria-current="page"' if s == current.slug else ''}>{esc(by_slug[s].nav)}</a></li>"
            for s in slugs
        )
        groups.append(f'<div class="nav-group"><h2>{esc(group)}</h2><ul>{links}</ul></div>')
    groups.append(
        '<div class="nav-group site-links"><h2>Cascade</h2><ul>'
        '<li><a href="../index.html">Home</a></li><li><a href="../writeup.html">Writeup</a></li>'
        f'<li><a href="{GITHUB}" target="_blank" rel="noopener">GitHub</a></li></ul></div>'
    )
    return "".join(groups)


def toc(headings: list[Heading]) -> str:
    if not headings:
        return ""
    items = "".join(
        f'<li class="l{h.level}"><a href="#{h.id}">{esc(h.text)}</a></li>' for h in headings
    )
    return f'<nav class="toc" aria-label="On this page"><h2>On this page</h2><ul>{items}</ul></nav>'


def pager(pages: list[Page], i: int) -> str:
    cells = []
    if i > 0:
        p = pages[i - 1]
        cells.append(f'<a class="prev" href="{p.href}"><span>Previous</span>{esc(p.nav)}</a>')
    else:
        cells.append("<span></span>")
    if i < len(pages) - 1:
        p = pages[i + 1]
        cells.append(f'<a class="next" href="{p.href}"><span>Next</span>{esc(p.nav)}</a>')
    return f'<nav class="pager" aria-label="Previous and next page">{"".join(cells)}</nav>'


def shell(page: Page, body: str, headings: list[Heading], pages: list[Page], i: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(page.title)} · Cascade docs</title>
<meta name="description" content="{html.escape(page.description)}">
<meta name="theme-color" content="#fafaf8" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#111110" media="(prefers-color-scheme: dark)">
<link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500&family=Geist+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="docs.css">
<script>try{{const t=localStorage.getItem("cascade-docs-theme");if(t)document.documentElement.dataset.theme=t}}catch{{}}</script>
<script src="search-index.js" defer></script>
<script src="docs.js" defer></script>
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="topbar">
  <button class="icon-btn menu-btn" type="button" aria-label="Open navigation" aria-expanded="false" aria-controls="sidebar">{ICONS["menu"]}</button>
  <a class="brand" href="index.html">{MARK}<span>Cascade</span><span class="brand-docs">docs</span></a>
  <button class="search-btn" type="button" aria-label="Search docs">{ICONS["search"]}<span>Search docs</span><kbd>/</kbd></button>
  <nav class="top-links" aria-label="Site">
    <a href="../index.html">Home</a>
    <a href="../writeup.html">Writeup</a>
    <a href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>
  </nav>
  <button class="icon-btn theme-btn" type="button" aria-label="Toggle dark mode">{ICONS["theme"]}</button>
</header>
<div class="layout">
  <aside class="sidebar" id="sidebar" aria-label="Docs">{sidebar(pages, page)}</aside>
  <main id="content" class="content">
    <p class="eyebrow">{esc(page.group)}</p>
    <h1>{esc(page.title)}</h1>
{body}
{pager(pages, i)}
    <footer class="doc-foot"><span>For AI agents</span><a href="agents.html">Use with AI agents</a><a href="agent-prompt.md">Agent prompt</a><a href="llms.txt">llms.txt</a><a href="llms-full.txt">llms-full.txt</a></footer>
  </main>
  {toc(headings)}
</div>
<div class="scrim" hidden></div>
<dialog class="search" aria-label="Search docs">
  <div class="search-box">{ICONS["search"]}<input type="search" placeholder="Search titles and headings" aria-label="Search" autocomplete="off" spellcheck="false"><kbd>esc</kbd></div>
  <ul class="results" role="listbox"></ul>
</dialog>
</body>
</html>
"""


def write(path: Path, content: str, written: set[Path]) -> None:
    written.add(path)
    written.update(p for p in path.parents if OUT in p.parents)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
        print(f"wrote {path.relative_to(HERE.parent.parent)}")


def main() -> int:
    check_type_notes()
    check_routes()
    pages = load_pages()
    OUT.mkdir(exist_ok=True)
    written: set[Path] = set()
    index = []
    bodies: dict[str, str] = {}
    for i, page in enumerate(pages):
        body, headings = render_body(page.source)
        bodies[page.slug] = body
        write(OUT / page.href, shell(page, body, headings, pages, i), written)
        index.append({"t": page.title, "u": page.href, "g": page.group})
        index += [{"t": h.text, "u": f"{page.href}#{h.id}", "g": page.title} for h in headings]
    write(
        OUT / "search-index.js",
        "window.DOCS_INDEX = " + json.dumps(index, ensure_ascii=False, indent=0) + ";\n",
        written,
    )
    write(OUT / "agent-prompt.md", agent_prompt(), written)
    write(OUT / "cascade-skill" / "SKILL.md", skill_file(), written)
    write(OUT / "llms.txt", llms_txt(pages), written)
    write(OUT / "llms-full.txt", llms_full(pages, bodies), written)
    for asset in ("docs.css", "docs.js"):
        write(OUT / asset, (HERE / asset).read_text(), written)
    for stale in set(OUT.iterdir()) - written:
        print(f"removed {stale.relative_to(HERE.parent.parent)}")
        shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
