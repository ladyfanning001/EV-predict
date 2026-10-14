# EV Battery Health API

API penilaian kesehatan baterai kendaraan listrik menggunakan model machine-learning.

## Prasyarat

- Python 3.11+
- Docker (opsional, untuk deployment via container)

## Cara Jalan Lokal

```bash
# 1. Buat virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependensi
pip install -r requirements.txt

# 3. Siapkan environment variable
cp .env.example .env
# Edit .env, ganti value BATTERY_API_KEY dengan key rahasia Anda

# 4. Jalankan server
export $(cat .env | xargs)
uvicorn main:app --reload --port 8000
```

Server akan berjalan di `http://localhost:8000`. Dokumentasi interaktif tersedia di `/docs`.

## Cara Test

```bash
# Pastikan BATTERY_API_KEY sudah diatur (atau biarkan test mengatur sendiri)
pip install pytest httpx
pytest tests/ -v
```

## Cara Build & Run Docker

```bash
# Build image
docker build -t ev-battery-api .

# Jalankan container
docker run -d \
  -e BATTERY_API_KEY=rahasia-anda \
  -e PORT=8000 \
  -p 8000:8000 \
  --name battery-api \
  ev-battery-api

# Test health check
curl http://localhost:8000/health
```

## Deploy ke Render

1. **Buat akun** di [render.com](https://render.com) dan hubungkan repo GitHub/GitLab Anda.
2. **Buat Web Service baru** → pilih repo ini.
3. **Konfigurasi build & start:**
   - **Environment**: Docker
   - **Region**: pilih yang terdekat dengan pengguna
4. **Environment Variables** (di dashboard Render):
   - `BATTERY_API_KEY` — key rahasia untuk autentikasi API
   - `PORT` — Render biasanya mengatur ini otomatis (default `8000`)
5. **Health Check Path**: atur ke `/health`
6. **Deploy** — Render akan build Docker image dan menjalankan container.

### Catatan

- Endpoint `/health` bersifat publik (tanpa API key) — cocok untuk health check Render.
- Endpoint `/assess` dan `/assess/batch` memerlukan header `x-api-key`.
- Pastikan file `battery_health_artifacts/` ikut ter-commit ke repo (berisi model).
