FROM python:3.10-slim

LABEL org.opencontainers.image.title="Data Preparation Pipeline Agent"
LABEL org.opencontainers.image.description="OpenEnv: 4-phase data prep — EDA, Cleaning, Engineering, Validation"
LABEL org.opencontainers.image.version="2.0.0"
LABEL space.tags="openenv"

WORKDIR /app

# Install system dependencies (critical for sklearn, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (important for dependency resolution)
RUN pip install --upgrade pip

# Copy entire project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Expose API port
EXPOSE 7860

# Health check (required)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]