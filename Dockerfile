FROM python:3.11-slim

# Buat user non-root
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

# Install dependencies terlebih dahulu (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy kode & artefak model
COPY . .

# Ganti ke user non-root
USER appuser

# PORT bisa di-override via env (default 8000)
ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
