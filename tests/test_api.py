"""Tes endpoint API Battery Health."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar 'main' bisa diimpor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setel env BATTERY_API_KEY sebelum import main (yang memvalidasi saat modul dimuat)
TEST_API_KEY = "test-secret-key-for-ci"
os.environ["BATTERY_API_KEY"] = TEST_API_KEY

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

VALID_HEADERS = {"x-api-key": TEST_API_KEY}


@pytest.fixture(scope="module")
def client():
    """TestClient sebagai context manager agar lifespan (model loading) berjalan."""
    with TestClient(app) as c:
        yield c


# ── /health ────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ── /assess – autentikasi ──────────────────────────────────────────────────
class TestAssessAuth:
    def test_no_key_returns_422(self, client):
        """FastAPI mengembalikan 422 bila header wajib tidak ada."""
        resp = client.post("/assess", json={"chemistry": "NMC"})
        assert resp.status_code == 422

    def test_wrong_key_returns_401(self, client):
        resp = client.post(
            "/assess",
            json={"chemistry": "NMC"},
            headers={"x-api-key": "wrong-key"},
        )
        assert resp.status_code == 401


# ── /assess – prediksi normal ─────────────────────────────────────────────
class TestAssessPredict:
    def test_valid_input_returns_passport(self, client):
        payload = {
            "chemistry": "NMC",
            "cycle_count": 900,
            "battery_age_days": 800,
        }
        resp = client.post("/assess", json=payload, headers=VALID_HEADERS)
        assert resp.status_code == 200
        data = resp.json()

        # Field wajib ada
        assert "grade" in data
        assert data["grade"] in ("A", "B", "C", "D")

        assert "state_of_health_pct" in data
        soh = data["state_of_health_pct"]
        assert "lower" in soh and "mid" in soh and "upper" in soh

        assert "remaining_life_cycles" in data
        rul = data["remaining_life_cycles"]
        assert "lower" in rul and "mid" in rul and "upper" in rul

        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_empty_body_returns_passport_with_warnings(self, client):
        """Input kosong ({}) tetap menghasilkan passport, dengan peringatan."""
        resp = client.post("/assess", json={}, headers=VALID_HEADERS)
        assert resp.status_code == 200
        data = resp.json()

        assert "grade" in data
        assert data["grade"] in ("A", "B", "C", "D")
        assert "warnings" in data
        assert isinstance(data["warnings"], list)
        # Dengan input kosong, harus ada peringatan kelengkapan
        assert len(data["warnings"]) > 0


# ── /assess/batch – batas 100 ─────────────────────────────────────────────
class TestAssessBatch:
    def test_batch_101_returns_422(self, client):
        items = [{"chemistry": "NMC"}] * 101
        resp = client.post(
            "/assess/batch",
            json={"batteries": items},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 422
