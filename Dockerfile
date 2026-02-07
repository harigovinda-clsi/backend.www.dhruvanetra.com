# DhruvaNetra Backend Container
# Built with patience, deployed with discipline

FROM python:3.11-slim

# Set working directory - our observatory base
WORKDIR /app

# Install system dependencies quietly
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching optimization)
COPY requirements.txt .

# Install Python dependencies with measured progress
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create non-root user for security (principle of least privilege)
RUN useradd -m -u 1000 observer && \
    chown -R observer:observer /app

USER observer

# Expose port (Render uses PORT environment variable)
ENV PORT=8000
EXPOSE 8000

# Health check - verify the eye remains open
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health')"

# Start the observatory
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info
