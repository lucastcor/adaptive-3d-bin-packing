FROM python:3.13-slim AS backend

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends g++ curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY cpp ./cpp
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .
RUN chmod +x scripts/build_native.sh && ./scripts/build_native.sh

EXPOSE 8000
CMD ["uvicorn", "adaptive_3dbp.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]

