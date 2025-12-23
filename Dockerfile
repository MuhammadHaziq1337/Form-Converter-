FROM python:3.10-slim

WORKDIR /app

# Copy only the Railway requirements
COPY requirements.railway.txt .

# Install dependencies
# Note: docling will be installed but its heavy ML model package (docling-ibm-models) is NOT included
RUN pip install --no-cache-dir -r requirements.railway.txt && \
    # Remove any heavy model files that might have been pulled in
    pip show docling-ibm-models > /dev/null 2>&1 && pip uninstall -y docling-ibm-models accelerate torch transformers || true

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
