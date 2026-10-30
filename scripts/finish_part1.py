"""
Resume and complete the remaining 6.9MB of Part 1, then assemble the wheel.
"""

import sys
import os
import socket
import time
import urllib.request
import ssl
from pathlib import Path

socket.setdefaulttimeout(20)

DEST_DIR = Path(__file__).resolve().parent / "wheels"
DEST_FILE = DEST_DIR / "pyside6_essentials-6.12.0-cp310-abi3-win_amd64.whl"
URL = "https://files.pythonhosted.org/packages/03/61/9a7e32a37b224f68506785bfb0f288d516098ec71681782d72b695a6c95d/pyside6_essentials-6.12.0-cp310-abi3-win_amd64.whl"
TOTAL_SIZE = 82153795

num_threads = 8
chunk_size = (TOTAL_SIZE + num_threads - 1) // num_threads

part_idx = 1
start_offset = part_idx * chunk_size
end_offset = (part_idx + 1) * chunk_size - 1
expected_total_len = end_offset - start_offset + 1

part_file = DEST_DIR / "pyside_part_1.tmp"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

current_sz = part_file.stat().st_size if part_file.exists() else 0
print(f"Part 1 current size: {current_sz} / {expected_total_len}", flush=True)

while current_sz < expected_total_len:
    req_start = start_offset + current_sz
    req_end = end_offset
    print(f"Requesting bytes {req_start}-{req_end} ({expected_total_len - current_sz} bytes remaining)...", flush=True)

    headers = {
        "Range": f"bytes={req_start}-{req_end}",
        "User-Agent": "Mozilla/5.0",
    }
    req = urllib.request.Request(URL, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp, open(part_file, "ab") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                current_sz += len(chunk)
                pct = (current_sz / expected_total_len) * 100
                sys.stdout.write(f"\r  Part 1 progress: {pct:.1f}% ({current_sz / 1_000_000:.2f}MB / {expected_total_len / 1_000_000:.2f}MB)")
                sys.stdout.flush()
        print()
    except Exception as e:
        print(f"\n[RETRY] Socket error: {e}, waiting 1s before resume...", flush=True)
        time.sleep(1)
        current_sz = part_file.stat().st_size

print("\nPart 1 complete! Assembling final wheel...", flush=True)

# Assemble all 8 parts
part_files = [DEST_DIR / f"pyside_part_{i}.tmp" for i in range(num_threads)]
tmp_assembly = DEST_FILE.with_suffix(".assembly")

with open(tmp_assembly, "wb") as outfile:
    for p_file in part_files:
        with open(p_file, "rb") as infile:
            while True:
                buf = infile.read(512 * 1024)
                if not buf:
                    break
                outfile.write(buf)

assembled_size = tmp_assembly.stat().st_size
if assembled_size == TOTAL_SIZE:
    if DEST_FILE.exists():
        DEST_FILE.unlink()
    tmp_assembly.rename(DEST_FILE)
    print(f"[SUCCESS] Wheel assembled successfully: {DEST_FILE.name} ({assembled_size / 1_000_000:.2f} MB)", flush=True)
    for p in part_files:
        try:
            p.unlink()
        except Exception:
            pass
else:
    print(f"[ERROR] Size mismatch: got {assembled_size}, expected {TOTAL_SIZE}", flush=True)
