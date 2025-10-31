import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class DepthAwareRefocusing:
    """
    Implements depth-aware refocusing effects.
    Simulates DSLR-style shallow depth of field based on depth maps.
    """
    
    def __init__(self, max_blur_radius=15, aperture_size=2.8):
        """
        Args:
            max_blur_radius (int): Maximum blur radius in pixels
            aperture_size (float): Simulated aperture f-stop (lower = more blur)
        """
        self.max_blur_radius = max_blur_radius
        self.aperture_size = aperture_size
    
    def compute_blur_map(self, depth_map, focus_point, focus_depth=None):
        """
        Compute blur amount for each pixel based on distance from focal plane.
        
        Args:
            depth_map (np.ndarray): Normalized depth map [0, 1]
            focus_point (tuple): (x, y) coordinates of focus point
            focus_depth (float): Depth at focus point (auto-computed if None)
            
        Returns:
            np.ndarray: Blur map with blur radius for each pixel
        """
        if focus_depth is None:
            # Get depth at focus point
            y, x = focus_point
            focus_depth = depth_map[y, x]
        
        # Compute depth difference from focal plane
        depth_diff = np.abs(depth_map - focus_depth)
        
        # Convert depth difference to blur radius
        # Further from focal plane = more blur
        blur_map = depth_diff * self.max_blur_radius / self.aperture_size
        
        # Clamp to maximum blur radius
        blur_map = np.clip(blur_map, 0, self.max_blur_radius)
        
        return blur_map.astype(np.float32)
    
    def apply_variable_blur(self, image, blur_map):
        """
        Apply variable blur to image based on blur map.
        
        Args:
            image (np.ndarray): Input RGB image (H, W, 3)
            blur_map (np.ndarray): Blur radius map (H, W)
            
        Returns:
            np.ndarray: Blurred image
        """
        # Create multiple blur kernels for different blur levels
        num_levels = 10
        max_blur = int(np.max(blur_map))
        
        if max_blur == 0:
            return image.copy()
        
        # Pre-compute blurred versions
        blurred_images = []
        blur_radii = np.linspace(0, max_blur, num_levels)
        
        for radius in blur_radii:
            if radius <= 0:
                blurred_images.append(image.copy())
            else:
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius, sigmaY=radius)
                blurred_images.append(blurred)
        
        # Blend between blur levels based on blur map
        result = np.zeros_like(image, dtype=np.float32)
        
        for i in range(len(blurred_images) - 1):
            # Create mask for current blur level
            lower_bound = blur_radii[i]
            upper_bound = blur_radii[i + 1]
            
            mask = ((blur_map >= lower_bound) & (blur_map < upper_bound)).astype(np.float32)
            
            if np.sum(mask) > 0:
                # Linear interpolation between blur levels
                alpha = (blur_map - lower_bound) / (upper_bound - lower_bound)
                alpha = np.clip(alpha, 0, 1)
                
                # Expand alpha to 3 channels
                alpha = np.expand_dims(alpha, axis=2)
                mask = np.expand_dims(mask, axis=2)
                
                # Blend images
                blended = (1 - alpha) * blurred_images[i] + alpha * blurred_images[i + 1]
                result += mask * blended
        
        # Handle maximum blur level
        max_mask = (blur_map >= blur_radii[-1]).astype(np.float32)
        max_mask = np.expand_dims(max_mask, axis=2)
        result += max_mask * blurred_images[-1]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def create_bokeh_effect(self, image, blur_map, bokeh_shape='circle'):
        """
        Create bokeh effect for out-of-focus highlights.
        
        Args:
            image (np.ndarray): Input RGB image
            blur_map (np.ndarray): Blur radius map
            bokeh_shape (str): Shape of bokeh ('circle', 'hexagon')
            
        Returns:
            np.ndarray: Image with bokeh effect
        """
        # Convert to LAB color space for better highlight detection
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness = lab[:, :, 0]
        
        # Detect highlights (bright areas)
        highlight_threshold = np.percentile(lightness, 90)
        highlights = lightness > highlight_threshold
        
        # Apply stronger blur to highlighted areas
        enhanced_blur_map = blur_map.copy()
        enhanced_blur_map[highlights] *= 1.5
        
        # Create custom bokeh kernel for very blurred areas
        large_blur_mask = enhanced_blur_map > (self.max_blur_radius * 0.7)
        
        if np.sum(large_blur_mask) > 0:
            # Create bokeh kernel
            kernel_size = int(self.max_blur_radius * 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            if bokeh_shape == 'circle':
                kernel = self._create_circular_kernel(kernel_size)
            else:  # hexagon
                kernel = self._create_hexagonal_kernel(kernel_size)
            
            # Apply bokeh blur to large blur areas
            for c in range(3):
                channel = image[:, :, c].astype(np.float32)
                bokeh_channel = cv2.filter2D(channel, -1, kernel)
                
                # Blend with original based on mask
                mask = large_blur_mask.astype(np.float32)
                image[:, :, c] = ((1 - mask) * channel + mask * bokeh_channel).astype(np.uint8)
        
        # Apply regular variable blur to the rest
        regular_blur_mask = ~large_blur_mask
        if np.sum(regular_blur_mask) > 0:
            regular_blur_map = blur_map.copy()
            regular_blur_map[large_blur_mask] = 0
            blurred = self.apply_variable_blur(image, regular_blur_map)
            
            # Combine results
            mask = regular_blur_mask.astype(np.float32)
            mask = np.expand_dims(mask, axis=2)
            image = ((1 - mask) * image + mask * blurred).astype(np.uint8)
        
        return image
    
    def _create_circular_kernel(self, size):
        """Create circular blur kernel for bokeh effect"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if distance <= center:
                    kernel[i, j] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        return kernel.astype(np.float32)
    
    def _create_hexagonal_kernel(self, size):
        """Create hexagonal blur kernel for bokeh effect"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Create hexagonal shape
        for i in range(size):
            for j in range(size):
                x, y = j - center, i - center
                # Hexagon equation: max(|x|, |y|, |x+y|) <= radius
                if max(abs(x), abs(y), abs(x + y)) <= center * 0.8:
                    kernel[i, j] = 1
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        return kernel.astype(np.float32)


class InteractiveRefocusing:
    """
    Handles interactive refocusing interface logic.
    """
    
    def __init__(self, max_blur_radius=15):
        self.refocuser = DepthAwareRefocusing(max_blur_radius=max_blur_radius)
        self.original_image = None
        self.depth_map = None
        self.current_result = None
        
    def set_image_and_depth(self, image, depth_map):
        """
        Set the input image and depth map for refocusing.
        
        Args:
            image (np.ndarray): RGB image (H, W, 3)
            depth_map (np.ndarray): Depth map (H, W)
        """
        self.original_image = image.copy()
        self.depth_map = depth_map.copy()
        self.current_result = image.copy()
    
    def refocus_at_point(self, x, y, aperture_size=2.8, bokeh_effect=True):
        """
        Apply refocusing effect at specified point.
        
        Args:
            x, y (int): Focus point coordinates
            aperture_size (float): Aperture size (lower = more blur)
            bokeh_effect (bool): Whether to apply bokeh effect
            
        Returns:
            np.ndarray: Refocused image
        """
        if self.original_image is None or self.depth_map is None:
            raise ValueError("Image and depth map must be set first")
        
        # Update aperture size
        self.refocuser.aperture_size = aperture_size
        
        # Ensure coordinates are within bounds
        h, w = self.depth_map.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Compute blur map
        blur_map = self.refocuser.compute_blur_map(self.depth_map, (y, x))
        
        # Apply refocusing
        if bokeh_effect:
            result = self.refocuser.create_bokeh_effect(self.original_image, blur_map)
        else:
            result = self.refocuser.apply_variable_blur(self.original_image, blur_map)
        
        self.current_result = result
        return result
    
    def get_depth_at_point(self, x, y):
        """Get depth value at specified coordinates"""
        if self.depth_map is None:
            return None
        
        h, w = self.depth_map.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        return float(self.depth_map[y, x])


def process_torch_depth(depth_tensor):
    """
    Convert PyTorch depth tensor to numpy for refocusing.
    
    Args:
        depth_tensor (torch.Tensor): Depth tensor from model
        
    Returns:
        np.ndarray: Processed depth map
    """
    if isinstance(depth_tensor, torch.Tensor):
        depth_np = depth_tensor.detach().cpu().numpy()
        
        # Handle batch dimension
        if len(depth_np.shape) == 4:  # (B, C, H, W)
            depth_np = depth_np[0, 0]  # Take first batch, first channel
        elif len(depth_np.shape) == 3:  # (C, H, W)
            depth_np = depth_np[0]  # Take first channel
    else:
        depth_np = depth_tensor
    
    # Normalize to [0, 1] if not already
    if depth_np.max() > 1.0:
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    
    return depth_np.astype(np.float32) 