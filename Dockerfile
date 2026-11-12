FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e '.[api]'
COPY scripts ./scripts
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && ./scripts/download.sh scifact && apt-get purge -y curl unzip && rm -rf /var/lib/apt/lists/*
ENV HRB_CORPUS=/app/data/scifact/corpus.jsonl
EXPOSE 8000
# BM25-only by default to keep the image small; add the [dense] extra and set HRB_DENSE_MODEL for hybrid.
CMD ["uvicorn", "hrb.api:app", "--host", "0.0.0.0", "--port", "8000"]
