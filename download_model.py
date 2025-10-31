#!/usr/bin/env python3
"""
Download model checkpoint from GitHub releases if not present locally.
This runs before the app starts in production.
"""
import os
import urllib.request
import sys

MODEL_PATH = "checkpoints/best_model.pth"
# Replace with your actual release URL
# Format: https://github.com/USERNAME/REPO/releases/download/TAG/FILENAME
MODEL_URL = (
    "https://github.com/bradymcatee/refocusing/releases/download/v1.0.0/best_model.pth"
)


def download_model():
    """Download model if it doesn't exist"""
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already exists at {MODEL_PATH}")
        return True

    print(f"Downloading model from {MODEL_URL}...")
    print("This may take a few minutes (1GB file)...")

    try:
        # Create checkpoints directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)

        # Download with progress
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress_hook)
        print("\n✓ Model downloaded successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False


if __name__ == "__main__":
    if not download_model():
        sys.exit(1)
