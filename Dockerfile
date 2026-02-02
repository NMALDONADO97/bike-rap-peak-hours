FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /tmp/requirements.txt

EXPOSE 8501

CMD ["bash", "-lc", "set -euo pipefail; \
  mkdir -p outputs reports data/raw data/processed; \
  levels=\"${BIKE_LEVEL:-hour day}\"; \
  for lvl in $levels; do \
    echo \"--- Running pipeline for level: $lvl ---\"; \
    BIKE_LEVEL=\"$lvl\" python src/pipeline.py; \
  done; \
  echo \"--- Launching Streamlit app ---\"; \
  exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --browser.serverAddress=localhost"]
