# Brand logos

Logos of the companies, model families, GPU vendors and tools the app names. `logos.json` lists each file and its source, and `_match` maps a Hugging Face model org or a GPU name prefix to a logo. `src/domain/logos.ts` reads it.

- `<key>.svg` is the full-color logo. `<key>-mono.svg` is single-color and is drawn as a mask in `currentColor`.
- Sources are pinned: `@lobehub/icons-static-svg@1.100.0` (MIT) and `simple-icons@13.21.0` (CC0). The Sail logo is cut from the wordmark on docs.sailresearch.com.
- The logos are trademarks of their owners and only refer to their products.
- `"tile": "mono"` marks a logo whose color version reads poorly on the white tile (Modal's light green), so the tile draws the mono one.
