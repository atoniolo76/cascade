"""Turn a rendered docs page body into Markdown, for llms-full.txt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

VOID = {"br", "hr", "img", "input", "meta", "link"}
BLOCK = {
    "p", "h1", "h2", "h3", "h4", "ul", "ol", "li", "div", "pre", "table", "figure",
    "blockquote", "section", "nav", "thead", "tbody", "tr",
}  # fmt: skip
SKIP = {"button", "script", "style", "svg", "dialog"}
FENCE_LANGS = {"bash": "bash", "python": "python", "js": "javascript", "text": "text"}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def find(self, cls: str) -> Node | None:
        for c in self.children:
            if isinstance(c, Node):
                if cls in c.classes:
                    return c
                hit = c.find(cls)
                if hit:
                    return hit
        return None

    def find_text(self, tag: str) -> str:
        hit = next(walk(self, tag), None)
        return inline(hit).strip() if hit else ""

    def text(self) -> str:
        return "".join(c if isinstance(c, str) else c.text() for c in self.children)


class Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: v or "" for k, v in attrs})
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, {k: v or "" for k, v in attrs}))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def fenced(code: str, lang: str = "") -> str:
    ticks = "`" * max(3, max((len(m) for m in re.findall(r"`+", code)), default=0) + 1)
    return f"{ticks}{lang}\n{code.rstrip()}\n{ticks}"


def inline(node: Node | str) -> str:
    if isinstance(node, str):
        return re.sub(r"\s+", " ", node)
    if node.tag in SKIP or "anchor" in node.classes:
        return ""
    inner = "".join(inline(c) for c in node.children)
    if node.tag in ("code", "kbd"):
        text = node.text()
        tick = "``" if "`" in text else "`"
        return f"{tick}{text}{tick}"
    if "constraint" in node.classes:
        return f"({inner.strip()})"
    if node.tag == "strong":
        return f"**{inner.strip()}**"
    if node.tag == "a" and node.attrs.get("href"):
        return f"[{inner.strip()}]({node.attrs['href']})"
    return inner


def cell(node: Node) -> str:
    return inline(node).strip().replace("|", "\\|")


def table(node: Node) -> str:
    rows = [
        [cell(td) for td in tr.children if isinstance(td, Node) and td.tag in ("td", "th")]
        for tr in walk(node, "tr")
    ]
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def walk(node: Node, tag: str):
    for c in node.children:
        if isinstance(c, Node):
            if c.tag == tag:
                yield c
            else:
                yield from walk(c, tag)


def code_div(node: Node) -> str:
    bar = node.find("code-bar")
    label = next((c.text().strip() for c in bar.children if isinstance(c, Node)), "") if bar else ""
    pre = next(walk(node, "pre"), None)
    code = pre.text() if pre else ""
    if label in FENCE_LANGS:
        return fenced(code, FENCE_LANGS[label])
    lang = "json" if "application/json" in label else ""
    return f"{label}:\n\n{fenced(code, lang)}" if label else fenced(code)


def list_block(node: Node, ordered: bool) -> str:
    items = []
    n = 0
    for li in node.children:
        if not (isinstance(li, Node) and li.tag == "li"):
            continue
        n += 1
        marker = f"{n}. " if ordered else "- "
        body = blocks(li)
        pad = " " * len(marker)
        lines = "\n\n".join(body).splitlines()
        items.append(
            marker + "\n".join(lines[:1] + [pad + line if line else "" for line in lines[1:]])
        )
    return "\n".join(items) if all("\n" not in i for i in items) else "\n\n".join(items)


def block(node: Node) -> list[str]:
    cls = node.classes
    if node.tag in SKIP or "anchor" in cls:
        return []
    if node.tag in ("h1", "h2", "h3", "h4"):
        return ["#" * int(node.tag[1]) + " " + inline(node).strip()]
    if node.tag == "p":
        text = inline(node).strip()
        return [text] if text else []
    if node.tag == "ul":
        return [list_block(node, ordered=False)]
    if node.tag == "ol":
        return [list_block(node, ordered=True)]
    if node.tag == "pre":
        return [fenced(node.text())]
    if node.tag == "table":
        return [table(node)]
    if node.tag == "figure":
        svg = next(walk(node, "svg"), None)
        caption = next(walk(node, "figcaption"), None)
        desc = (svg.attrs.get("aria-label") if svg else None) or (caption.text() if caption else "")
        return [f"Diagram: {desc.strip()}"] if desc else []
    if node.tag == "blockquote" or cls & {"note", "warn"}:
        inner = "\n\n".join(blocks(node))
        return ["\n".join(f"> {line}" if line else ">" for line in inner.splitlines())]
    if "code" in cls:
        return [code_div(node)]
    if "route" in cls:
        parts = [c.text().strip() for c in node.children if isinstance(c, Node)]
        return [f"`{parts[0]} {parts[1]}` ({parts[2]})"]
    if "cards" in cls:
        cards = [a for a in node.children if isinstance(a, Node) and a.tag == "a"]
        return [
            "\n".join(
                f"- [{next(walk(a, 'strong')).text()}]({a.attrs['href']}): {a.find_text('span')}"
                for a in cards
            )
        ]
    if "tab-list" in cls:
        return []
    return blocks(node)


def blocks(node: Node) -> list[str]:
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        text = "".join(run).strip()
        if text:
            out.append(text)
        run.clear()

    for c in node.children:
        if isinstance(c, Node) and (c.tag in BLOCK or c.tag == "figure"):
            flush()
            out.extend(b for b in block(c) if b)
        else:
            run.append(inline(c))
    flush()
    return out


def to_markdown(html_body: str, page: str) -> str:
    """`page` is the page's file name, which fragment-only links resolve against."""
    tree = Tree()
    tree.feed(html_body.replace('href="#', f'href="{page}#'))
    tree.close()
    return "\n\n".join(blocks(tree.root)) + "\n"
