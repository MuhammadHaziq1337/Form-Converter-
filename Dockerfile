FROM python:3.10-slim

WORKDIR /app

# Copy only the Railway requirements
COPY requirements.railway.txt .

# Install lightweight dependencies first
RUN pip install --no-cache-dir -r requirements.railway.txt

# Install docling WITHOUT optional ML dependencies
# This gives us DocumentConverter but skips torch/transformers/cuda
RUN pip install --no-cache-dir --no-deps docling==2.65.0

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
