"""Inline results/<dataset>/explorer_data.json into docs/index.html (static page, no server, no network).

Usage: python scripts/build_explorer.py [--datasets california king_county]
Build with a single dataset (for example ``--datasets california``) to leave the other out of the page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = "/*__DATA__*/null"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["california", "king_county"])
    args = ap.parse_args()
    template = (ROOT / "scripts" / "explorer_template.html").read_text(encoding="utf-8")
    data = {}
    for name in args.datasets:
        path = ROOT / "results" / name / "explorer_data.json"
        if path.exists():
            data[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"skipping {name}: {path} not found")
    if not data:
        raise SystemExit("no results found; run train and evaluate first")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    assert MARK in template
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(template.replace(MARK, payload), encoding="utf-8")
    print(f"wrote {out} with {sorted(data)} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
