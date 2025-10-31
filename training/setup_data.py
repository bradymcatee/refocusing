#!/usr/bin/env python3
"""
Setup script for NYU Depth v2 dataset.
Downloads and prepares the dataset for training.
"""

import os
import urllib.request
import zipfile
from pathlib import Path


def download_with_progress(url, filename):
    """Download file with progress bar"""

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        bar_length = 50
        filled_length = int(bar_length * percent / 100)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        print(
            f"\rDownloading: |{bar}| {percent:.1f}% ({downloaded}/{total_size} bytes)",
            end="",
        )

    urllib.request.urlretrieve(url, filename, progress_hook)
    print()  # New line after progress bar


def setup_nyu_depth_v2(data_dir="./data/nyu_depth_v2"):
    """Setup NYU Depth v2 dataset"""

    print("NYU Depth v2 Dataset Setup")
    print("=" * 50)

    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_path.absolute()}")

    # Check if dataset already exists
    dataset_file = data_path / "nyu_depth_v2_labeled.mat"

    if dataset_file.exists():
        print(f"Dataset already exists at: {dataset_file}")
        file_size = dataset_file.stat().st_size / (1024**3)  # Size in GB
        print(f"File size: {file_size:.2f} GB")
        return str(dataset_file)

    print("\nDataset Download Instructions:")
    print("-" * 30)
    print("The NYU Depth v2 dataset is ~2.8GB and needs to be downloaded manually.")
    print()
    print("Steps:")
    print("1. Go to: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
    print("2. Download 'Labeled dataset (~2.8 GB)' - nyu_depth_v2_labeled.mat")
    print(f"3. Place the file in: {data_path.absolute()}")
    print()
    print("Alternative direct download:")
    print(
        "wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    print()

    # Try automatic download
    download_url = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )

    try:
        print(f"Attempting automatic download from: {download_url}")
        print("This may take a while (2.8GB file)...")

        download_with_progress(download_url, str(dataset_file))

        print(f"Dataset downloaded successfully to: {dataset_file}")

        # Verify file size
        file_size = dataset_file.stat().st_size / (1024**3)
        print(f"File size: {file_size:.2f} GB")

        if file_size < 2.5:  # Expected size is ~2.8GB
            print("Warning: Downloaded file seems smaller than expected")
            print("Please verify the download completed successfully")

        return str(dataset_file)

    except Exception as e:
        print(f"Automatic download failed: {e}")
        print("Please download manually using the instructions above.")
        return None


def verify_dataset(data_dir="./data/nyu_depth_v2"):
    """Verify the dataset can be loaded"""
    dataset_file = Path(data_dir) / "nyu_depth_v2_labeled.mat"

    if not dataset_file.exists():
        print("Dataset file not found. Please run setup first.")
        return False

    try:
        import h5py
        import numpy as np

        print("Verifying dataset...")

        with h5py.File(dataset_file, "r") as f:
            print("Dataset structure:")
            for key in f.keys():
                dataset = f[key]
                print(f"  {key}: {dataset.shape} ({dataset.dtype})")

            # Check if we can read a sample
            images = np.array(f["images"])
            depths = np.array(f["depths"])

            print(f"\nImages shape: {images.shape}")
            print(f"Depths shape: {depths.shape}")
            print(f"Number of samples: {images.shape[-1]}")

        print("Dataset verification successful!")
        return True

    except ImportError:
        print("h5py not installed. Please install: pip install h5py")
        return False
    except Exception as e:
        print(f"Dataset verification failed: {e}")
        return False


def test_dataloader(data_dir="./data/nyu_depth_v2"):
    """Test the dataloader with a small sample"""
    try:
        from nyu_dataset import NYUDataLoader

        print("Testing dataloader...")  # Create a small test dataloader
        data_loader = NYUDataLoader(data_dir, batch_size=2, num_workers=0)
        train_loader, val_loader, test_loader = data_loader.get_dataloaders()

        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")

        # Test loading a batch
        sample_batch = next(iter(train_loader))
        print(f"RGB batch shape: {sample_batch['rgb'].shape}")
        print(f"Depth batch shape: {sample_batch['depth'].shape}")

        print("Dataloader test successful!")
        return True

    except Exception as e:
        print(f"Dataloader test failed: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup NYU Depth v2 Dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/nyu_depth_v2",
        help="Directory to store dataset",
    )
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset")
    parser.add_argument("--test", action="store_true", help="Test dataloader")

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.data_dir)
    elif args.test:
        test_dataloader(args.data_dir)
    else:
        # Setup dataset
        result = setup_nyu_depth_v2(args.data_dir)

        if result:
            print("\nNext steps:")
            print("1. Verify dataset: python setup_data.py --verify")
            print("2. Test dataloader: python setup_data.py --test")
            print("3. Start training: python train.py --data_path ./data/nyu_depth_v2")


if __name__ == "__main__":
    main()
