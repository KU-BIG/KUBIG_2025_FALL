# CLAP2Diffusion Docker Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding large files)
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY configs/ ./configs/
COPY utils/ ./utils/

# Create necessary directories
RUN mkdir -p /app/outputs /app/temp /app/logs /app/checkpoints /app/data

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose port for Gradio
EXPOSE 7860

# Note: checkpoints and data should be mounted via volumes, not copied

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app/gradio_app.py"]