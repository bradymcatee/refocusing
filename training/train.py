import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
from tqdm import tqdm
import numpy as np

from src.models.multiscale_cnn import create_model
from nyu_dataset import NYUDataLoader


class DepthLoss(nn.Module):
    """
    Combined loss function for depth estimation.
    Includes L2 loss and gradient loss for better edge preservation.
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super(DepthLoss, self).__init__()
        self.alpha = alpha  # Weight for L2 loss
        self.beta = beta  # Weight for gradient loss
        self.l2_loss = nn.MSELoss()

    def gradient_loss(self, pred, target):
        """Compute gradient loss to preserve edges"""
        # Compute gradients using Sobel filters
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        sobel_x = sobel_x.view(1, 1, 3, 3).to(pred.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(pred.device)

        # Compute gradients
        pred_grad_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1)
        target_grad_x = torch.nn.functional.conv2d(target, sobel_x, padding=1)
        target_grad_y = torch.nn.functional.conv2d(target, sobel_y, padding=1)

        # Compute gradient magnitude
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)

        return self.l2_loss(pred_grad_mag, target_grad_mag)

    def forward(self, pred, target):
        # L2 loss
        l2 = self.l2_loss(pred, target)

        # Gradient loss
        grad = self.gradient_loss(pred, target)

        return self.alpha * l2 + self.beta * grad


class Trainer:
    """Training manager for the multi-scale CNN"""

    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args

        # Loss function
        self.criterion = DepthLoss(alpha=0.7, beta=0.3)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )

        # Logging
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.best_val_loss = float("inf")
        self.step = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        coarse_loss_total = 0
        fine_loss_total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            rgb = batch["rgb"].to(self.device)
            depth_target = batch["depth"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(rgb)
            coarse_depth = outputs["coarse_depth"]
            fine_depth = outputs["fine_depth"]

            # Resize targets for coarse network
            coarse_target = torch.nn.functional.interpolate(
                depth_target,
                size=coarse_depth.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

            # Compute losses
            coarse_loss = self.criterion(coarse_depth, coarse_target)
            fine_loss = self.criterion(fine_depth, depth_target)

            # Combined loss (weighted toward fine network)
            loss = 0.3 * coarse_loss + 0.7 * fine_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            coarse_loss_total += coarse_loss.item()
            fine_loss_total += fine_loss.item()

            # Log to tensorboard
            if self.step % self.args.log_interval == 0:
                self.writer.add_scalar("Train/Loss", loss.item(), self.step)
                self.writer.add_scalar(
                    "Train/CoarseLoss", coarse_loss.item(), self.step
                )
                self.writer.add_scalar("Train/FineLoss", fine_loss.item(), self.step)
                self.writer.add_scalar(
                    "Train/LearningRate",
                    self.optimizer.param_groups[0]["lr"],
                    self.step,
                )

            self.step += 1

            # Update progress bar with more precision
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.6f}",
                    "Coarse": f"{coarse_loss.item():.6f}",
                    "Fine": f"{fine_loss.item():.6f}",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        avg_coarse_loss = coarse_loss_total / len(self.train_loader)
        avg_fine_loss = fine_loss_total / len(self.train_loader)

        return avg_loss, avg_coarse_loss, avg_fine_loss

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        coarse_loss_total = 0
        fine_loss_total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                rgb = batch["rgb"].to(self.device)
                depth_target = batch["depth"].to(self.device)

                # Forward pass
                outputs = self.model(rgb)
                coarse_depth = outputs["coarse_depth"]
                fine_depth = outputs["fine_depth"]

                # Resize targets
                coarse_target = torch.nn.functional.interpolate(
                    depth_target,
                    size=coarse_depth.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

                # Compute losses
                coarse_loss = self.criterion(coarse_depth, coarse_target)
                fine_loss = self.criterion(fine_depth, depth_target)
                loss = 0.3 * coarse_loss + 0.7 * fine_loss

                total_loss += loss.item()
                coarse_loss_total += coarse_loss.item()
                fine_loss_total += fine_loss.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_coarse_loss = coarse_loss_total / len(self.val_loader)
        avg_fine_loss = fine_loss_total / len(self.val_loader)

        # Log validation metrics
        self.writer.add_scalar("Val/Loss", avg_loss, epoch)
        self.writer.add_scalar("Val/CoarseLoss", avg_coarse_loss, epoch)
        self.writer.add_scalar("Val/FineLoss", avg_fine_loss, epoch)

        # Save sample predictions
        if epoch % self.args.save_sample_interval == 0:
            self.save_sample_predictions(
                rgb[:4], depth_target[:4], fine_depth[:4], epoch
            )

        return avg_loss, avg_coarse_loss, avg_fine_loss

    def save_sample_predictions(self, rgb, target, pred, epoch):
        """Save sample predictions to tensorboard"""
        # Denormalize RGB images for visualization
        rgb_vis = rgb.cpu().clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb_vis = rgb_vis * std + mean
        rgb_vis = torch.clamp(rgb_vis, 0, 1)

        # Log images
        self.writer.add_images("Samples/RGB", rgb_vis, epoch)
        self.writer.add_images("Samples/DepthTarget", target, epoch)
        self.writer.add_images("Samples/DepthPred", pred, epoch)

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "step": self.step,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {loss:.4f}")

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.args.epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_coarse, train_fine = self.train_epoch(epoch)

            # Validate
            val_loss, val_coarse, val_fine = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{self.args.epochs} - {epoch_time:.2f}s")
            print(
                f"Train Loss: {train_loss:.4f} (Coarse: {train_coarse:.4f}, Fine: {train_fine:.4f})"
            )
            print(
                f"Val Loss: {val_loss:.4f} (Coarse: {val_coarse:.4f}, Fine: {val_fine:.4f})"
            )
            print("-" * 60)

        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Multi-Scale CNN for Depth Estimation"
    )

    # Data arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to NYU Depth v2 dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (reduced for memory efficiency)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loading workers (reduced for stability)",
    )

    # Model arguments
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[304, 228],
        help="Target image size [width, height]",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=15, help="Step size for learning rate decay"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.5, help="Gamma for learning rate decay"
    )

    # Logging and saving
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for tensorboard logs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval for logging training metrics",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5, help="Interval for saving checkpoints"
    )
    parser.add_argument(
        "--save_sample_interval",
        type=int,
        default=2,
        help="Interval for saving sample predictions",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Memory optimization tips
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
        )

        # Clear cache and optimize memory usage
        torch.cuda.empty_cache()

        if args.batch_size > 8:
            print(
                "WARNING: Large batch size detected. Consider reducing if you encounter memory issues."
            )

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load data
    print("Loading dataset...")
    data_loader = NYUDataLoader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=tuple(args.target_size),
    )

    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print("Creating model...")
    model = create_model()
    model.to(device)

    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, device, args)
    trainer.train()


if __name__ == "__main__":
    main()
