FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application files
COPY . .

# Download model on build
RUN python download_model.py

# Expose port (Hugging Face uses port 7860)
EXPOSE 7860

# Run the application
CMD gunicorn --bind 0.0.0.0:7860 --workers 1 --timeout 300 app:app
