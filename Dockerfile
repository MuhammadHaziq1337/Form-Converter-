FROM python:3.10-slim

WORKDIR /app

# Copy only the Railway requirements
COPY requirements.railway.txt .

# Step 1: Install CPU-only PyTorch FIRST (saves ~3GB by preventing CUDA downloads)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install all lightweight dependencies (including docling-ibm-models which now gets CPU torch)
RUN pip install --no-cache-dir -r requirements.railway.txt

# Step 3: Install docling WITH its dependencies (torch-cpu already in place prevents CUDA download)
RUN pip install --no-cache-dir docling==2.65.0

# Step 4: Explicitly uninstall any CUDA/GPU packages that snuck in
RUN pip uninstall -y nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
    nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-cufile-cu12 \
    accelerate || true

# Copy the entire app
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "server.py"]
