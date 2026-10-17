"""Download the 40 CWRU recordings used here, plus the 24 locked confirmatory recordings,
into data/cwru/ with size checks and retries.

    python scripts/download_cwru.py

Writes data/cwru/MANIFEST.json with each file's size and SHA-256 as downloaded, so a
later run can be checked against it. The data belong to Case Western Reserve
University's Bearing Data Center; they are fetched, not redistributed, by this repo.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sustain_iq.cwru_files import FILES, LOCKED_FILES, URL  # noqa: E402

OUT = ROOT / "data" / "cwru"


def size_on_server(url: str) -> int | None:
    for _ in range(4):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(r.headers["Content-Length"])
        except Exception:  # noqa: BLE001 - network flakiness; retry
            time.sleep(5)
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    failed = []
    for fault, diameter, load, n, label, *_ in [*FILES, *LOCKED_FILES]:
        dest = OUT / f"{label}.mat"
        url = URL.format(n=n)
        want = size_on_server(url)
        if dest.exists() and want is not None and dest.stat().st_size == want:
            print(f"ok (have)   {label:12s} file {n}")
        else:
            ok = False
            for attempt in range(1, 6):
                try:
                    with urllib.request.urlopen(url, timeout=120) as r:
                        data = r.read()
                    if want is not None and len(data) != want:
                        raise IOError(f"size {len(data)} != {want}")
                    dest.write_bytes(data)
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  retry {label} (attempt {attempt}): {e}")
                    time.sleep(5)
            if not ok:
                failed.append(label)
                continue
            print(f"downloaded  {label:12s} file {n}  {dest.stat().st_size} bytes")
        manifest[label] = {"file": n, "bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\n{len(manifest)}/{len(FILES) + len(LOCKED_FILES)} files present; failed: {failed or 'none'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
