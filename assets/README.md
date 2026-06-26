# assets/

Master imagery for IcefallDB. The two SVGs are the canonical sources — every
other file here (and any reproduction elsewhere) is a derived raster that must
trace back to them. The vectors are mirrored from the IcefallDB brand assets in
the [`icefalldb.com`](https://github.com/visorcraft/icefalldb.com) repo; both
wordmark and mark use outlined `<path>` glyphs (no `<text>`, no web-font
dependency), so they render identically everywhere — including GitHub — and a
naive raster export is faithful.

| File | Size | Purpose |
| ---- | ---- | ------- |
| `icefalldb-logo.svg` | scalable | Source-of-truth **wordmark** (icon + `IcefallDB` + `CHILLINGLY FLEXIBLE`). Render to any size with `rsvg-convert -w <px>`. |
| `icefalldb-logo.png` | 2048×583 | Wordmark master raster — transparent, for docs, slide decks, and consumers that can't read SVG. |
| `icefalldb-mark.svg` | scalable | Source-of-truth **mark** — the icon only, no text. |
| `icefalldb-mark.png` | 1024×1024 | Mark master raster — transparent. Used as the README hero (the SVG's soft-shadow `<filter>` is stripped by GitHub's sanitizer, so the PNG renders the glow faithfully). |
| `icefalldb.ico` | 16/32/48/64/128/256 | Multi-resolution icon built from the mark, for favicons and tooling that prefers `.ico`. |
| `social-1280x640.png` | 1280×640 | GitHub social preview / OpenGraph card. Upload via **Settings → Social preview** on github.com. |

## Regenerating from the SVGs

```sh
# Wordmark + mark master rasters.
rsvg-convert -w 2048 assets/icefalldb-logo.svg -o assets/icefalldb-logo.png
rsvg-convert -w 1024 -h 1024 assets/icefalldb-mark.svg -o assets/icefalldb-mark.png

# Multi-resolution ICO from the mark.
magick assets/icefalldb-mark.png \
  -define icon:auto-resize=256,128,64,48,32,16 assets/icefalldb.ico
```

The `social-1280x640.png` card is hand-composed: the wordmark
(`icefalldb-logo.png`) centered over the site's diagonal brand gradient
(`#020713` → `#06101d` → `#071d32`) with a one-line caption. Regenerate it with
the brand toolkit (numpy + Pillow) if the wordmark or tagline changes.
