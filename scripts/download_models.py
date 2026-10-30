"""
PhantomTouch Model Downloader
Downloads official Google MediaPipe task models for Hand Landmarker,
Face Landmarker, and Image Segmenter.
"""

import sys
import os
import urllib.request
import ssl
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Official MediaPipe model URLs and minimum expected sizes (bytes)
MODELS = {
    "hand_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "alt_url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "min_size": 5_000_000,  # ~8-10MB
        "desc": "Hand Landmarker (float16)",
    },
    "face_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "alt_url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "min_size": 3_000_000,  # ~3.76MB
        "desc": "Face Landmarker (float16)",
    },
    "selfie_segmenter.tflite": {
        "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        "alt_url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
        "min_size": 100_000,  # ~250KB
        "desc": "Selfie Segmenter (float16)",
    },
}


def download_file(url: str, dest_path: Path, min_size: int, desc: str) -> bool:
    """Download a file with progress reporting and validation."""
    print(f"Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    # Create unverified context if corporate SSL inspection blocks certs
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    temp_path = dest_path.with_suffix(".tmp")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PhantomTouch/1.0"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=60) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 64 * 1024  # 64 KB

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {percent:6.1f}% ({mb_down:5.1f} MB / {mb_total:5.1f} MB)")
                    sys.stdout.flush()

        print()  # newline
        downloaded_size = temp_path.stat().st_size
        if downloaded_size < min_size:
            print(f"  [ERROR] File too small ({downloaded_size} bytes, expected >= {min_size} bytes).")
            if temp_path.exists():
                temp_path.unlink()
            return False

        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        print(f"  [SUCCESS] Saved {dest_path.name} ({downloaded_size / (1024 * 1024):.2f} MB)")
        return True

    except Exception as e:
        print(f"\n  [ERROR] Failed to download {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def ensure_models(force: bool = False) -> bool:
    """Ensure all required model files are present and valid."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_success = True

    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        if not force and dest.exists():
            size = dest.stat().st_size
            if size >= info["min_size"]:
                print(f"[OK] {filename} exists ({size / (1024 * 1024):.2f} MB)")
                continue
            else:
                print(f"[WARN] {filename} is corrupted or incomplete ({size} bytes). Re-downloading...")

        # Try primary URL, then alt URL
        success = download_file(info["url"], dest, info["min_size"], info["desc"])
        if not success and "alt_url" in info:
            print("  Retrying with alternate URL...")
            success = download_file(info["alt_url"], dest, info["min_size"], info["desc"])

        if not success:
            all_success = False

    return all_success


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    print("=" * 60)
    print("PhantomTouch - MediaPipe Tasks Vision Model Downloader")
    print(f"Target directory: {MODELS_DIR}")
    print("=" * 60)

    ok = ensure_models(force=force_download)
    if ok:
        print("\nAll models are ready!")
        sys.exit(0)
    else:
        print("\n[FAILED] One or more models failed to download.")
        print("Please check your internet connection and re-run.")
        sys.exit(1)
