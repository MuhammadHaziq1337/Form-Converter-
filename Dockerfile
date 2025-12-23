FROM python:3.13-slim

WORKDIR /app

# Copy only the lightweight requirements
COPY requirements.railway.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.railway.txt

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
