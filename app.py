#!/usr/bin/env python3
"""
Interactive Refocusing Web Application
"""

from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

from src.models.multiscale_cnn import create_model
from src.utils.refocusing import InteractiveRefocusing, process_torch_depth

# Try to download model if it doesn't exist
try:
    from download_model import download_model

    download_model()
except Exception as e:
    print(f"Note: Could not run download_model: {e}")

# Initialize Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
refocuser = InteractiveRefocusing(max_blur_radius=20)


def load_model():
    """Load the trained model"""
    global model
    checkpoint_path = "./checkpoints/best_model.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return False

    try:
        model = create_model()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def image_to_base64(image_array):
    """Convert numpy array to base64 string for web display"""
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    """Handle image upload and depth prediction"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        # Read and process image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array
        image_np = np.array(image)

        # Predict depth
        depth_map = predict_depth(image_np)

        # Setup refocuser
        refocuser.set_image_and_depth(image_np, depth_map)

        # Create depth visualization
        depth_normalized = (
            (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        ).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        # Convert to base64 for web display
        original_b64 = image_to_base64(image_np)
        depth_b64 = image_to_base64(depth_colored)

        return jsonify(
            {
                "success": True,
                "original_image": f"data:image/png;base64,{original_b64}",
                "depth_map": f"data:image/png;base64,{depth_b64}",
                "image_shape": image_np.shape[:2],  # height, width
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/refocus", methods=["POST"])
def refocus_image():
    """Handle refocusing request"""
    try:
        data = request.get_json()

        # Get parameters
        x = int(data["x"])
        y = int(data["y"])
        aperture = float(data.get("aperture", 2.8))
        bokeh = data.get("bokeh", True)

        # Perform refocusing
        refocused = refocuser.refocus_at_point(
            x, y, aperture_size=aperture, bokeh_effect=bokeh
        )

        # Convert to base64
        refocused_b64 = image_to_base64(refocused)

        # Get focus depth value
        focus_depth = (
            refocuser.depth_map[y, x] if refocuser.depth_map is not None else 0
        )

        return jsonify(
            {
                "success": True,
                "refocused_image": f"data:image/png;base64,{refocused_b64}",
                "focus_depth": float(focus_depth),
                "focus_point": [x, y],
                "aperture": aperture,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def predict_depth(image):
    """Predict depth map from image"""
    if model is None:
        raise Exception("Model not loaded")

    # Preprocess image
    image_resized = cv2.resize(image, (304, 228))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Predict depth
    with torch.no_grad():
        depth_pred = model.predict_depth(image_tensor)
        depth_np = process_torch_depth(depth_pred)

        # Resize to original image size
        depth_map = cv2.resize(depth_np, (image.shape[1], image.shape[0]))

    return depth_map


@app.route("/status")
def status():
    """Get application status"""
    return jsonify(
        {
            "model_loaded": model is not None,
            "device": str(device),
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
        }
    )


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# Load model when module is imported (works with gunicorn)
print("Initializing model...")
if not load_model():
    print("WARNING: Model not loaded. App may not work correctly.")
    print("Make sure checkpoints/best_model.pth exists.")
else:
    print(f"Model loaded on device: {device}")


if __name__ == "__main__":
    print("Starting Interactive Refocusing Web App")
    print("=" * 50)

    # Load model
    if not load_model():
        print("Failed to load model. Please check checkpoints/best_model.pth")
        exit(1)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print("\nStarting web server...")
    print("Access the app at: http://localhost:5000")
    print("Ready for deployment!")

    # Run the app
    app.run(
        host="0.0.0.0",  # Allow external connections
        port=5000,
        debug=False,  # Set to True for development
    )
