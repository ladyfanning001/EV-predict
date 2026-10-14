"""EV Battery Health Assessment API.

Menyediakan endpoint untuk menilai kesehatan baterai EV menggunakan
model machine-learning yang sudah dilatih.
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from battery_health_inference import BatteryHealthPredictor

# ---------------------------------------------------------------------------
# Konfigurasi & startup
# ---------------------------------------------------------------------------
BATTERY_API_KEY: str = os.environ.get("BATTERY_API_KEY", "")
if not BATTERY_API_KEY:
    raise RuntimeError(
        "Environment variable BATTERY_API_KEY belum diatur. "
        "Setel variabel ini sebelum menjalankan aplikasi. "
        "Contoh: export BATTERY_API_KEY=rahasia-anda"
    )

predictor: BatteryHealthPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Muat model sekali saat startup."""
    global predictor
    predictor = BatteryHealthPredictor("battery_health_artifacts/battery_health_bundle.joblib")
    yield


app = FastAPI(
    title="EV Battery Health API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Autentikasi
# ---------------------------------------------------------------------------
def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """Bandingkan header x-api-key dengan env BATTERY_API_KEY."""
    if not secrets.compare_digest(x_api_key, BATTERY_API_KEY):
        raise HTTPException(status_code=401, detail="API key tidak valid.")
    return x_api_key


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class BatteryInput(BaseModel):
    """Input satu baterai. Semua field opsional agar input parsial tetap bisa diproses."""
    model_config = {"extra": "allow"}


class BatchRequest(BaseModel):
    """Batch input: list of battery dicts, maksimal 100."""
    batteries: list[dict[str, Any]]

    @field_validator("batteries")
    @classmethod
    def max_100(cls, v: list) -> list:
        if len(v) > 100:
            raise ValueError("Batch maksimal 100 item.")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health-check publik. Tidak membutuhkan API key."""
    return {"status": "ok"}


@app.post("/assess")
async def assess(request: Request, _key: str = Depends(verify_api_key)):
    """Nilai satu baterai; mengembalikan battery passport."""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body harus berupa JSON object.")
    passport = predictor.predict(body)
    return passport


@app.post("/assess/batch")
async def assess_batch(batch: BatchRequest, _key: str = Depends(verify_api_key)):
    """Nilai beberapa baterai sekaligus (maks 100)."""
    passports = predictor.predict(batch.batteries)
    return {"results": passports}
