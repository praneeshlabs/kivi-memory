FROM python:3.11-slim
WORKDIR /app

# CPU-only torch keeps the image ~800MB instead of ~4GB with CUDA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model into the image so cold starts don't download it.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')"

ENV DATABASE_PATH=/data/kivi.db
EXPOSE 8000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
