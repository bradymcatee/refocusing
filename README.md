# Interactive Refocusing with Monocular Depth Estimation

A web application that simulates DSLR-style shallow focus effects from single RGB images using CNN-based depth estimation. Built with Eigen et al.'s Multi-Scale CNN architecture for computational photography.

![Interactive Refocusing Demo](https://img.shields.io/badge/Status-Complete-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)

## Features

- **Deep Learning**: Multi-Scale CNN for monocular depth estimation
- **Web Interface**: Interactive Flask application with modern UI
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Interaction**: Click-to-focus with live preview
- **Visual Feedback**: Animated focus points and depth visualization
- **Configurable**: Adjustable aperture and bokeh effects
- **Deployable**: Ready for production deployment

## Architecture

### Model Architecture

- **Coarse Network**: Global scene structure (304×228 → 74×55)
- **Fine Network**: Detail refinement with multi-scale features
- **Parameters**: 87.9M trainable parameters
- **Training**: NYU Depth v2 dataset (~50K RGB-D pairs)

### Performance Metrics

- **RMSE**: 0.1787
- **MAE**: 0.1393
- **SSIM**: 0.7140
- **δ-accuracy**: ~79% (δ < 1.25³)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model Exists

Make sure you have the trained model at:

```
checkpoints/best_model.pth
```

### 3. Run the App

```bash
# Development mode (with auto-reload)
python run.py

# Production mode (requires: pip install gunicorn)
python run.py --prod

# Custom host/port
python run.py --host 0.0.0.0 --port 8080
```

Open your browser to http://localhost:5000

## Web Interface Usage

1. **Upload Image**: Drag & drop or click to select an image
2. **Wait for Processing**: Automatic depth prediction using CNN
3. **Interactive Focus**: Click anywhere on the image to refocus
4. **Adjust Settings**:
   - Aperture slider (f/1.4 - f/8.0)
   - Bokeh effect toggle
5. **Download Results**: Save refocused images

## Usage

### Running the Web App

**Development Mode** (with hot-reload):

```bash
python run.py
```

**Production Mode** (with gunicorn):

```bash
pip install gunicorn  # if not already installed
python run.py --prod --host 0.0.0.0 --port 5000 --workers 4
```

### Standalone Script

You can also run the app directly:

```bash
python app.py
```

## Project Structure

```
refocusing/
├── run.py                      # Main launcher script
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # Web interface
├── src/
│   ├── models/
│   │   └── multiscale_cnn.py  # CNN architecture
│   └── utils/
│       └── refocusing.py      # Refocusing algorithms
├── training/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Model evaluation
│   ├── setup_data.py          # Dataset setup
│   ├── nyu_dataset.py         # Dataset loader
│   └── data/                  # Downloaded dataset files
│       └── nyu_depth_v2/
├── checkpoints/
│   └── best_model.pth         # Trained model
└── demo_outputs/              # Example outputs
```

## API Endpoints

### `POST /upload`

Upload and process image for depth estimation

- **Input**: Multipart form with image file
- **Output**: JSON with base64-encoded original and depth images

### `POST /refocus`

Apply refocusing at specified coordinates

- **Input**: JSON with x, y coordinates, aperture, bokeh settings
- **Output**: JSON with base64-encoded refocused image

### `GET /status`

Check application and model status

- **Output**: JSON with model status, device info, GPU availability

## Technical Details

### Depth Estimation

- Input resolution: 304×228 (optimized for training)
- Output resolution: Full image resolution via upsampling
- Normalization: ImageNet preprocessing
- Inference time: ~100ms on RTX 2070

### Refocusing Algorithm

- **Depth-aware blur**: Variable Gaussian kernels based on depth
- **Circle of confusion**: Realistic aperture simulation
- **Bokeh rendering**: Enhanced highlight effects
- **Multi-scale processing**: Preserves fine details

### Web Technologies

- **Backend**: Flask with async support
- **Frontend**: Bootstrap 5 + vanilla JavaScript
- **Image handling**: Base64 encoding for web display
- **Real-time updates**: AJAX with loading indicators

## Training (Optional)

To retrain the model from scratch:

1. **Install training dependencies**:

```bash
# Uncomment and install training dependencies in requirements.txt
pip install matplotlib h5py scipy tqdm tensorboard
```

2. **Download NYU Depth v2 dataset**:

```bash
cd training
python setup_data.py
```

3. **Train the model**:

```bash
python training/train.py --epochs 100 --batch-size 16
```

Training uses Adam optimizer, L2 + gradient loss, and requires ~8GB VRAM.

## Deployment

For production deployment, use gunicorn:

```bash
pip install gunicorn
python run.py --prod --host 0.0.0.0 --port 5000 --workers 4
```

For Docker deployment, create a simple `Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

## References

- Eigen, D., Puhrsch, C., & Fergus, R. (2014). Depth map prediction from a single image using a multi-scale deep network. NIPS.
- Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012). Indoor segmentation and support inference from RGBD images. ECCV.

## Academic Use

This implementation is designed for educational and research purposes. If you use this code in your research, please cite:

```bibtex
@inproceedings{eigen2014depth,
  title={Depth map prediction from a single image using a multi-scale deep network},
  author={Eigen, David and Puhrsch, Christian and Fergus, Rob},
  booktitle={Advances in neural information processing systems},
  pages={2366--2374},
  year={2014}
}
```

## Future Enhancements

- [ ] Real-time video processing
- [ ] Mobile app development (React Native)
- [ ] Advanced bokeh shapes (hexagonal, etc.)
- [ ] Multi-person depth estimation
- [ ] Integration with modern architectures (DPT, MiDaS)
- [ ] WebGL acceleration for client-side processing

---
