"""
Robust multi-threaded segmented downloader with per-chunk files and retries.
"""

import sys
import os
import time
import urllib.request
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEST_DIR = Path(__file__).resolve().parent / "wheels"
DEST_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://files.pythonhosted.org/packages/03/61/9a7e32a37b224f68506785bfb0f288d516098ec71681782d72b695a6c95d/pyside6_essentials-6.12.0-cp310-abi3-win_amd64.whl"
DEST_FILE = DEST_DIR / "pyside6_essentials-6.12.0-cp310-abi3-win_amd64.whl"
TOTAL_SIZE = 82153795

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_part(part_idx, start, end, part_file):
    expected_len = end - start + 1
    # Check if part already fully downloaded
    if part_file.exists() and part_file.stat().st_size == expected_len:
        print(f"Part {part_idx + 1} already cached.", flush=True)
        return part_idx, expected_len

    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "Mozilla/5.0",
    }

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=40) as resp, open(part_file, "wb") as f:
                downloaded = 0
                while True:
                    buf = resp.read(64 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)

            if downloaded == expected_len:
                print(f"[OK] Part {part_idx + 1} downloaded ({downloaded / 1_000_000:.2f} MB)", flush=True)
                return part_idx, downloaded
            else:
                print(f"[RETRY] Part {part_idx + 1}: expected {expected_len}, got {downloaded}", flush=True)
        except Exception as e:
            print(f"[RETRY] Part {part_idx + 1} attempt {attempt} error: {e}", flush=True)
            time.sleep(1)

    raise RuntimeError(f"Part {part_idx + 1} failed after {max_attempts} attempts.")


def main():
    if DEST_FILE.exists() and DEST_FILE.stat().st_size == TOTAL_SIZE:
        print(f"[OK] {DEST_FILE.name} already exists and valid.", flush=True)
        return

    num_threads = 8
    chunk_size = (TOTAL_SIZE + num_threads - 1) // num_threads

    print(f"Starting parallel download: {num_threads} parts for {TOTAL_SIZE / 1_000_000:.1f} MB...", flush=True)
    futures = []
    part_files = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = min(TOTAL_SIZE - 1, (i + 1) * chunk_size - 1)
            p_file = DEST_DIR / f"pyside_part_{i}.tmp"
            part_files.append(p_file)
            futures.append(executor.submit(download_part, i, start, end, p_file))

        for f in as_completed(futures):
            f.result()

    print("All parts downloaded! Assembling final wheel...", flush=True)
    tmp_assembled = DEST_FILE.with_suffix(".assembly")
    with open(tmp_assembled, "wb") as outfile:
        for p_file in part_files:
            with open(p_file, "rb") as infile:
                while True:
                    buf = infile.read(512 * 1024)
                    if not buf:
                        break
                    outfile.write(buf)

    if tmp_assembled.stat().st_size == TOTAL_SIZE:
        if DEST_FILE.exists():
            DEST_FILE.unlink()
        tmp_assembled.rename(DEST_FILE)
        # Clean up temporary part files
        for p_file in part_files:
            try:
                p_file.unlink()
            except Exception:
                pass
        print(f"[SUCCESS] Saved final wheel: {DEST_FILE.name} ({TOTAL_SIZE / 1_000_000:.2f} MB)", flush=True)
    else:
        print(f"[ERROR] Size mismatch: got {tmp_assembled.stat().st_size}, expected {TOTAL_SIZE}", flush=True)


if __name__ == "__main__":
    main()
