import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseNetwork(nn.Module):
    """
    Coarse scale network that captures global scene structure.
    Implements the first stage of Eigen et al.'s multi-scale architecture.
    """
    
    def __init__(self):
        super(CoarseNetwork, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc6 = nn.Linear(256 * 6 * 8, 4096)  # Adjusted for 304x228 input
        self.fc7 = nn.Linear(4096, 4096)
        
        # Output layer for coarse depth prediction
        self.fc8 = nn.Linear(4096, 74 * 55)  # 74x55 coarse output
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Store intermediate activations for fine network
        activations = {}
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        activations['pool1'] = x
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        activations['pool2'] = x
        
        x = F.relu(self.conv3(x))
        activations['conv3'] = x
        
        x = F.relu(self.conv4(x))
        activations['conv4'] = x
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        activations['pool5'] = x
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected processing
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        activations['fc6'] = x
        
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        activations['fc7'] = x
        
        # Output coarse depth map
        coarse_depth = self.fc8(x)
        coarse_depth = coarse_depth.view(-1, 1, 74, 55)  # Reshape to spatial dimensions
        
        return coarse_depth, activations


class FineNetwork(nn.Module):
    """
    Fine scale network that refines depth details using multi-scale features.
    Second stage of Eigen et al.'s architecture.
    """
    
    def __init__(self):
        super(FineNetwork, self).__init__()
        
        # Fine-scale convolutional layers
        self.conv1 = nn.Conv2d(3, 63, kernel_size=9, stride=2, padding=4)
        self.conv2 = nn.Conv2d(63, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        
        # Combine with coarse prediction
        self.conv4 = nn.Conv2d(64 + 1, 64, kernel_size=5, stride=1, padding=2)  # +1 for coarse depth
        self.conv5 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)
        
    def forward(self, rgb_image, coarse_depth):
        # Process RGB image at finer scale
        x = F.relu(self.conv1(rgb_image))  # 304x228 -> 152x114
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Upsample coarse depth to match fine features
        coarse_upsampled = F.interpolate(coarse_depth, size=(x.size(2), x.size(3)), 
                                       mode='bilinear', align_corners=True)
        
        # Concatenate fine features with coarse depth
        x = torch.cat([x, coarse_upsampled], dim=1)
        
        # Final refinement
        x = F.relu(self.conv4(x))
        fine_depth = self.conv5(x)
        
        # Upsample to full resolution
        fine_depth = F.interpolate(fine_depth, size=(rgb_image.size(2), rgb_image.size(3)), 
                                 mode='bilinear', align_corners=True)
        
        return fine_depth


class MultiScaleCNN(nn.Module):
    """
    Complete multi-scale CNN for depth estimation.
    Combines coarse and fine networks as described in Eigen et al.
    """
    
    def __init__(self):
        super(MultiScaleCNN, self).__init__()
        self.coarse_net = CoarseNetwork()
        self.fine_net = FineNetwork()
        
    def forward(self, x):
        # Get coarse depth prediction and intermediate features
        coarse_depth, activations = self.coarse_net(x)
        
        # Refine with fine network
        fine_depth = self.fine_net(x, coarse_depth)
        
        return {
            'coarse_depth': coarse_depth,
            'fine_depth': fine_depth,
            'activations': activations
        }
    
    def predict_depth(self, x):
        """Convenience method for inference"""
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['fine_depth']


def create_model():
    """Factory function to create the multi-scale CNN model"""
    return MultiScaleCNN()


if __name__ == "__main__":
    # Test model architecture
    model = create_model()
    
    # Test with dummy input (batch_size=2, channels=3, height=304, width=228)
    dummy_input = torch.randn(2, 3, 304, 228)
    
    print("Testing MultiScale CNN...")
    outputs = model(dummy_input)
    
    print(f"Coarse depth shape: {outputs['coarse_depth'].shape}")
    print(f"Fine depth shape: {outputs['fine_depth'].shape}")
    print("Model test successful!") 