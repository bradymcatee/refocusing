# Model Checkpoint

This directory should contain the trained model file: `best_model.pth`

## Download

The model file is too large for GitHub. Download it separately:

**Option 1:** Google Drive / Dropbox

- Upload your `best_model.pth` to cloud storage
- Share the link in the main README

**Option 2:** GitHub Releases

- Create a release and attach the .pth file
- GitHub allows files up to 2GB in releases

**Option 3:** Hugging Face Hub

```bash
# Upload to Hugging Face
pip install huggingface-hub
huggingface-cli upload your-username/refocusing ./checkpoints/best_model.pth
```

## File Info

- Size: ~1 GB
- Format: PyTorch checkpoint (.pth)
- Architecture: Multi-Scale CNN for depth estimation
