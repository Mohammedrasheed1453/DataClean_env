FROM python:3.10-slim

LABEL org.opencontainers.image.title="Data Preparation Pipeline Agent"
LABEL org.opencontainers.image.description="OpenEnv: 4-phase data prep — EDA, Cleaning, Engineering, Validation"
LABEL org.opencontainers.image.version="2.0.0"
LABEL space.tags="openenv"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py            .
COPY dataset_generator.py .
COPY client.py            .
COPY inference.py         .
COPY grader.py            .
COPY openenv.yaml         .
COPY __init__.py          .
COPY server/              ./server/

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]