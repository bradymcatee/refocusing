---
title: Interactive Refocusing
emoji: 📷
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Interactive Refocusing with Monocular Depth Estimation

A web application that simulates DSLR-style shallow focus effects from single RGB images using CNN-based depth estimation.

## How to Use

1. Upload an RGB image
2. View the predicted depth map
3. Click anywhere on the image to refocus at that depth
4. Adjust aperture size (f/1.4 - f/8.0)
5. Toggle bokeh effects on/off
6. Download your refocused images

## Model

Multi-Scale CNN trained on NYU Depth v2 dataset for monocular depth estimation.
