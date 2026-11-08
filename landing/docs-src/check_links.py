"""Check every internal link in landing/docs/: run `python3 landing/docs-src/check_links.py`.

Crawls from docs/index.html, follows every relative href and src, and fails on a missing file or
a #fragment that names no id on its target page. Also checks the Markdown links in llms.txt and
llms-full.txt the same way.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag

DOCS = Path(__file__).resolve().parent.parent / "docs"
MARKDOWN_INDEXES = ("llms.txt", "llms-full.txt")
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.refs: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if a.get(key):
                self.refs.append(a[key])


_cache: dict[Path, Links] = {}


def parse(path: Path) -> Links:
    if path not in _cache:
        _cache[path] = Links()
        _cache[path].feed(path.read_text())
    return _cache[path]


def check(page: Path, ref: str, errors: list[str]) -> Path | None:
    """Records a broken ref and returns the docs page it points at, if any."""
    url, frag = urldefrag(ref)
    target = (page.parent / unquote(url)).resolve() if url else page
    if not target.exists():
        errors.append(f"{page.name}: {ref} -> missing file")
        return None
    if frag and target.suffix == ".html" and unquote(frag) not in parse(target).ids:
        errors.append(f"{page.name}: {ref} -> no id {frag!r}")
    return target if target.suffix == ".html" and target.parent == DOCS else None


def internal(ref: str) -> bool:
    return not ref.startswith(("http://", "https://", "mailto:"))


def main() -> int:
    parsed: dict[Path, Links] = {}
    queue, errors, checked = [DOCS / "index.html"], [], 0
    while queue:
        page = queue.pop()
        if page in parsed:
            continue
        parsed[page] = parse(page)
        for ref in filter(internal, parsed[page].refs):
            checked += 1
            target = check(page, ref, errors)
            if target:
                queue.append(target)
    for name in MARKDOWN_INDEXES:
        index = DOCS / name
        for ref in filter(internal, MD_LINK.findall(index.read_text())):
            checked += 1
            check(index, ref, errors)
    pages = {p for p in parsed if p.parent == DOCS}
    orphans = set(DOCS.glob("*.html")) - pages
    errors += [f"{p.name}: no page links here" for p in sorted(orphans)]
    for e in errors:
        print(e)
    print(f"{len(pages)} pages, {checked} internal links and anchors checked, {len(errors)} broken")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
