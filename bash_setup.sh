#!/bin/bash

# Update package index
sudo apt update

# Install Python 3, pip, and venv (if not already installed)
sudo apt install -y python3 python3-pip python3-venv

# Create a new virtual environment
python3 -m venv ocr_env

# Activate the virtual environment
source ocr_env/bin/activate

# Upgrade pip inside the venv
pip install --upgrade pip

pip install  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PaddlePaddle
pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install PaddleOCR and dependencies
pip install "paddleocr>=2.0.1"

# Install Flask and Gunicorn
pip install flask gunicorn

# Test imports (optional)
python3 -c "import paddleocr; import torch; print('✅ PaddleOCR and PyTorch are ready!')"
