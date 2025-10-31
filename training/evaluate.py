import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from src.models.multiscale_cnn import create_model
from nyu_dataset import NYUDataLoader


class DepthMetrics:
    """Compute depth estimation metrics"""

    @staticmethod
    def rmse(pred, target):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((pred - target) ** 2))

    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        return np.mean(np.abs(pred - target))

    @staticmethod
    def delta_accuracy(pred, target, threshold=1.25):
        """Delta accuracy (percentage of pixels within threshold ratio)"""
        ratio = np.maximum(pred / target, target / pred)
        return np.mean(ratio < threshold) * 100

    @staticmethod
    def log_rmse(pred, target):
        """Log Root Mean Squared Error"""
        return np.sqrt(np.mean((np.log(pred + 1e-8) - np.log(target + 1e-8)) ** 2))

    @staticmethod
    def abs_rel(pred, target):
        """Absolute Relative Error"""
        return np.mean(np.abs(pred - target) / target)

    @staticmethod
    def sq_rel(pred, target):
        """Squared Relative Error"""
        return np.mean(((pred - target) ** 2) / target)


def evaluate_model(
    model, test_loader, device, save_samples=True, output_dir="./evaluation"
):
    """Evaluate the model on test set"""
    model.eval()

    metrics = {
        "rmse": [],
        "mae": [],
        "delta_1": [],  # δ < 1.25
        "delta_2": [],  # δ < 1.25²
        "delta_3": [],  # δ < 1.25³
        "log_rmse": [],
        "abs_rel": [],
        "sq_rel": [],
        "ssim": [],
    }

    sample_count = 0
    max_samples_to_save = 10

    print("Evaluating model...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluation")):
            rgb = batch["rgb"].to(device)
            depth_target = batch["depth"].to(device)

            # Forward pass
            outputs = model(rgb)
            depth_pred = outputs["fine_depth"]

            # Convert to numpy for metrics computation
            pred_np = depth_pred.cpu().numpy()
            target_np = depth_target.cpu().numpy()

            # Compute metrics for each sample in batch
            for i in range(pred_np.shape[0]):
                pred_sample = pred_np[i, 0]  # Remove channel dimension
                target_sample = target_np[i, 0]

                # Ensure positive values for ratio-based metrics
                pred_sample = np.maximum(pred_sample, 1e-8)
                target_sample = np.maximum(target_sample, 1e-8)

                # Compute metrics
                metrics["rmse"].append(DepthMetrics.rmse(pred_sample, target_sample))
                metrics["mae"].append(DepthMetrics.mae(pred_sample, target_sample))
                metrics["delta_1"].append(
                    DepthMetrics.delta_accuracy(pred_sample, target_sample, 1.25)
                )
                metrics["delta_2"].append(
                    DepthMetrics.delta_accuracy(pred_sample, target_sample, 1.25**2)
                )
                metrics["delta_3"].append(
                    DepthMetrics.delta_accuracy(pred_sample, target_sample, 1.25**3)
                )
                metrics["log_rmse"].append(
                    DepthMetrics.log_rmse(pred_sample, target_sample)
                )
                metrics["abs_rel"].append(
                    DepthMetrics.abs_rel(pred_sample, target_sample)
                )
                metrics["sq_rel"].append(
                    DepthMetrics.sq_rel(pred_sample, target_sample)
                )

                # SSIM
                ssim_score = ssim(target_sample, pred_sample, data_range=1.0)
                metrics["ssim"].append(ssim_score)

                # Save sample predictions
                if save_samples and sample_count < max_samples_to_save:
                    save_sample_prediction(
                        rgb[i].cpu(),
                        target_sample,
                        pred_sample,
                        sample_count,
                        output_dir,
                    )
                    sample_count += 1

    # Compute average metrics
    avg_metrics = {}
    for key, values in metrics.items():
        avg_metrics[key] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)

    return avg_metrics


def save_sample_prediction(rgb, target, pred, sample_idx, output_dir):
    """Save a sample prediction visualization"""
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize RGB image
    rgb_np = rgb.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_np = rgb_np * std + mean
    rgb_np = np.clip(rgb_np, 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RGB image
    axes[0].imshow(rgb_np)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Ground truth depth
    im1 = axes[1].imshow(target, cmap="plasma")
    axes[1].set_title("Ground Truth Depth")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Predicted depth
    im2 = axes[2].imshow(pred, cmap="plasma")
    axes[2].set_title("Predicted Depth")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"sample_{sample_idx:03d}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def print_metrics(metrics):
    """Print evaluation metrics in a formatted table"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12}")
    print("-" * 40)

    # Core metrics
    core_metrics = ["rmse", "mae", "abs_rel", "sq_rel", "log_rmse"]
    for metric in core_metrics:
        mean_val = metrics[metric]
        std_val = metrics[f"{metric}_std"]
        print(f"{metric.upper():<15} {mean_val:<12.4f} {std_val:<12.4f}")

    print()
    print("Delta Accuracy (%):")
    delta_metrics = ["delta_1", "delta_2", "delta_3"]
    for metric in delta_metrics:
        mean_val = metrics[metric]
        std_val = metrics[f"{metric}_std"]
        threshold = ["1.25", "1.25²", "1.25³"][delta_metrics.index(metric)]
        print(f"δ < {threshold:<8} {mean_val:<12.2f} {std_val:<12.2f}")

    print()
    print(f"SSIM Score:     {metrics['ssim']:<12.4f} {metrics['ssim_std']:<12.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Scale CNN for Depth Estimation"
    )

    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to NYU Depth v2 dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[304, 228],
        help="Target image size [width, height]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation",
        help="Output directory for sample predictions",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save sample prediction visualizations",
    )
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

    # Load model
    print("Loading model...")
    model = create_model()

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Model loaded from {args.checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'Unknown')}")

    # Load test data
    print("Loading test dataset...")
    data_loader = NYUDataLoader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=tuple(args.target_size),
    )

    _, _, test_loader = data_loader.get_dataloaders()
    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate model
    metrics = evaluate_model(
        model,
        test_loader,
        device,
        save_samples=args.save_samples,
        output_dir=args.output_dir,
    )

    # Print results
    print_metrics(metrics)

    # Save metrics to file
    import json

    metrics_file = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_file}")

    if args.save_samples:
        print(f"Sample predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
