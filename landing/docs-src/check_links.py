"""Check every internal link in landing/docs/: run `python3 landing/docs-src/check_links.py`.

Crawls from docs/index.html, follows every relative href and src, and fails on a missing file or
a #fragment that names no id on its target page.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag

DOCS = Path(__file__).resolve().parent.parent / "docs"


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


def main() -> int:
    parsed: dict[Path, Links] = {}
    queue, errors, checked = [DOCS / "index.html"], [], 0
    while queue:
        page = queue.pop()
        if page in parsed:
            continue
        parsed[page] = parse(page)
        for ref in parsed[page].refs:
            if ref.startswith(("http://", "https://", "mailto:")):
                continue
            url, frag = urldefrag(ref)
            target = (page.parent / unquote(url)).resolve() if url else page
            checked += 1
            if not target.exists():
                errors.append(f"{page.name}: {ref} -> missing file")
                continue
            if target.suffix == ".html" and target.parent == DOCS:
                queue.append(target)
            if frag:
                if unquote(frag) not in parse(target).ids:
                    errors.append(f"{page.name}: {ref} -> no id {frag!r}")
    pages = {p for p in parsed if p.parent == DOCS}
    orphans = set(DOCS.glob("*.html")) - pages
    errors += [f"{p.name}: no page links here" for p in sorted(orphans)]
    for e in errors:
        print(e)
    print(f"{len(pages)} pages, {checked} internal links and anchors checked, {len(errors)} broken")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
