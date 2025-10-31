import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os


class NYUDepthDataset(Dataset):
    """
    NYU Depth v2 dataset loader for training the multi-scale CNN.
    Handles RGB images and corresponding depth maps.
    Memory-efficient implementation that loads samples on-demand.
    """
    
    def __init__(self, data_path, split='train', transform=None, target_size=(304, 228)):
        """
        Args:
            data_path (str): Path to NYU Depth v2 dataset
            split (str): 'train', 'val', or 'test'
            transform: Optional transform to apply to images
            target_size (tuple): Target size for resizing images (width, height)
        """
        self.data_path = data_path
        self.split = split
        self.target_size = target_size
        
        # Store dataset file path for on-demand loading
        self.dataset_file = os.path.join(self.data_path, 'nyu_depth_v2_labeled.mat')
        
        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(
                f"NYU Depth v2 dataset not found at {self.dataset_file}. "
                "Please download from: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"
            )
        
        # Get dataset info and compute depth statistics
        self.setup_dataset_info()
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size[::-1]),  # PIL expects (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            self.transform = transform
            
        # For depth, use only resizing - no PIL conversion to avoid precision loss
        self.depth_transform = transforms.Resize(target_size[::-1], interpolation=Image.NEAREST)
    
    def setup_dataset_info(self):
        """Get dataset information and compute depth statistics for proper normalization"""
        print(f"Setting up dataset info from {self.dataset_file}...")
        
        with h5py.File(self.dataset_file, 'r') as f:
            # Get dataset dimensions without loading data
            images_shape = f['images'].shape  # (1449, 3, 640, 480)
            depths_shape = f['depths'].shape  # (1449, 640, 480)
            
            # Number of samples is the first dimension
            total_samples = images_shape[0]
            
            print(f"Dataset contains {total_samples} samples")
            print(f"Image shape: {images_shape}")
            print(f"Depth shape: {depths_shape}")
            
            # Compute depth statistics for better normalization
            print("Computing depth statistics for normalization...")
            # Sample a few depth maps to get statistics
            sample_indices = np.linspace(0, total_samples-1, min(50, total_samples), dtype=int)
            depth_values = []
            
            for idx in sample_indices:
                depth_sample = f['depths'][idx, :, :]
                valid_depths = depth_sample[depth_sample > 0]  # Remove invalid values
                depth_values.append(valid_depths)
            
            all_depths = np.concatenate(depth_values)
            self.depth_min = np.percentile(all_depths, 1)  # 1st percentile
            self.depth_max = np.percentile(all_depths, 99)  # 99th percentile
            
            print(f"Computed depth range: [{self.depth_min:.4f}, {self.depth_max:.4f}] meters")
        
        # Split dataset indices
        if self.split == 'train':
            self.indices = list(range(int(0.8 * total_samples)))
        elif self.split == 'val':
            self.indices = list(range(int(0.8 * total_samples), int(0.9 * total_samples)))
        else:  # test
            self.indices = list(range(int(0.9 * total_samples), total_samples))
        
        print(f"{self.split} split: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single sample - loads data on demand"""
        actual_idx = self.indices[idx]
        
        # Open file and load only the specific sample we need
        with h5py.File(self.dataset_file, 'r') as f:
            # Load single RGB image: shape (1449, 3, 640, 480) -> select [actual_idx, :, :, :]
            rgb_data = f['images'][actual_idx, :, :, :]  # Shape: (3, 640, 480)
            rgb_image = np.transpose(rgb_data, (1, 2, 0))  # Shape: (640, 480, 3)
            
            # Load single depth map: shape (1449, 640, 480) -> select [actual_idx, :, :]
            depth_map = f['depths'][actual_idx, :, :]  # Shape: (640, 480)
        
        # Convert RGB to PIL for transforms
        rgb_pil = Image.fromarray(rgb_image)
        rgb_tensor = self.transform(rgb_pil)
        
        # Process depth WITHOUT PIL to avoid precision loss
        # 1. Normalize depth using computed statistics
        depth_clipped = np.clip(depth_map, self.depth_min, self.depth_max)
        depth_normalized = (depth_clipped - self.depth_min) / (self.depth_max - self.depth_min)
        
        # 2. Convert to tensor and resize
        depth_tensor = torch.from_numpy(depth_normalized).float().unsqueeze(0)  # Add channel dim
        depth_resized = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0), 
            size=self.target_size[::-1], 
            mode='bilinear', 
            align_corners=True
        ).squeeze(0)  # Remove batch dim
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_resized,
            'original_depth': torch.tensor(depth_map, dtype=torch.float32)
        }


class NYUDataLoader:
    """Convenience wrapper for creating train/val/test dataloaders"""
    
    def __init__(self, data_path, batch_size=32, num_workers=4, target_size=(304, 228)):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
    
    def get_dataloaders(self):
        """Return train, validation, and test dataloaders"""
        
        # Create datasets
        train_dataset = NYUDepthDataset(
            self.data_path, split='train', target_size=self.target_size
        )
        val_dataset = NYUDepthDataset(
            self.data_path, split='val', target_size=self.target_size
        )
        test_dataset = NYUDepthDataset(
            self.data_path, split='test', target_size=self.target_size
        )
        
        # Create dataloaders with persistent workers for better performance
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return train_loader, val_loader, test_loader


def download_nyu_depth_v2(data_path):
    """
    Helper function to download NYU Depth v2 dataset.
    Note: This is a placeholder - you'll need to manually download the dataset.
    """
    print("NYU Depth v2 Dataset Download Instructions:")
    print("1. Go to: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
    print("2. Download 'Labeled dataset (~2.8 GB)' - nyu_depth_v2_labeled.mat")
    print(f"3. Place the file in: {data_path}")
    print("4. The dataset contains 1449 pairs of RGB and depth images")
    
    # Create directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)


if __name__ == "__main__":
    # Test dataset loading
    data_path = "./data/nyu_depth_v2"
    
    if not os.path.exists(os.path.join(data_path, 'nyu_depth_v2_labeled.mat')):
        download_nyu_depth_v2(data_path)
    else:
        # Test dataloader
        print("Testing NYU Depth Dataset...")
        
        try:
            data_loader = NYUDataLoader(data_path, batch_size=4, num_workers=0)
            train_loader, val_loader, test_loader = data_loader.get_dataloaders()
            
            # Test loading a batch
            sample_batch = next(iter(train_loader))
            print(f"RGB batch shape: {sample_batch['rgb'].shape}")
            print(f"Depth batch shape: {sample_batch['depth'].shape}")
            print("Dataset test successful!")
            
        except Exception as e:
            print(f"Error testing dataset: {e}")
            download_nyu_depth_v2(data_path) 