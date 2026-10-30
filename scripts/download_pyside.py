"""
Download PySide6-Essentials and Shiboken6 wheels with streaming chunks.
"""

import os
import sys
import urllib.request
import ssl
from pathlib import Path

DEST_DIR = Path(__file__).resolve().parent / "wheels"
DEST_DIR.mkdir(parents=True, exist_ok=True)

WHEELS = {
    "shiboken6-6.12.0-cp310-abi3-win_amd64.whl": "https://files.pythonhosted.org/packages/2a/b6/bee97e7a2edeab72aafabb46ee5adc939790a638d107860b4cad7a20e407/shiboken6-6.12.0-cp310-abi3-win_amd64.whl",
    "pyside6_essentials-6.12.0-cp310-abi3-win_amd64.whl": "https://files.pythonhosted.org/packages/03/61/9a7e32a37b224f68506785bfb0f288d516098ec71681782d72b695a6c95d/pyside6_essentials-6.12.0-cp310-abi3-win_amd64.whl",
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for filename, url in WHEELS.items():
    dest = DEST_DIR / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[OK] {filename} already exists ({dest.stat().st_size / 1_000_000:.1f} MB)")
        continue

    print(f"Downloading {filename}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temp = dest.with_suffix(".tmp")
    with urllib.request.urlopen(req, context=ctx, timeout=180) as resp, open(temp, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        done = 0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total > 0:
                pct = (done / total) * 100
                sys.stdout.write(f"\r  {pct:.1f}% ({done / 1_000_000:.1f}MB / {total / 1_000_000:.1f}MB)")
                sys.stdout.flush()
    print()
    temp.rename(dest)
    print(f"[SUCCESS] Saved {dest.name}")

print("All wheels ready in:", DEST_DIR)
