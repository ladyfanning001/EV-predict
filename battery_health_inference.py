"""Battery health inference: grade A-D, SOH & RUL intervals, confidence, passport.

Dipakai oleh notebook (training/evaluasi) DAN oleh web/API (inference), jadi
logika prediksinya cuma ada satu di sini.

Contoh:
    from battery_health_inference import BatteryHealthPredictor
    predictor = BatteryHealthPredictor("artifacts/battery_health_bundle.joblib")
    passport = predictor.predict({"chemistry": "NMC", "cycle_count": 820, ...})
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

GRADE_ORDER = ["A", "B", "C", "D"]
DEFAULT_THRESHOLDS = {"A": 80.0, "B": 70.0, "C": 60.0}  # D = di bawah 60
MIN_COMPLETENESS = 0.8  # di bawah ini, keyakinan dibatasi kelengkapan input


# --------------------------------------------------------------------------- #
# Fungsi inti (dipakai saat training & inference)
# --------------------------------------------------------------------------- #
def soh_to_grade(soh, thresholds=None):
    """SOH (%) -> grade A/B/C/D."""
    t = thresholds or DEFAULT_THRESHOLDS
    soh = np.asarray(soh, dtype=float)
    return np.select(
        [soh >= t["A"], soh >= t["B"], soh >= t["C"]], ["A", "B", "C"], default="D"
    )


def raw_quantiles(models, X):
    """Prediksi kuantil bawah/tengah/atas; diurutkan agar tidak 'crossing'."""
    lo, mid, hi = (models[k].predict(X) for k in ("lo", "mid", "hi"))
    s = np.sort(np.vstack([lo, mid, hi]), axis=0)
    return s[0], s[1], s[2]


def predict_interval(models, offset, X, lower=0.0, upper=None):
    """Interval hasil conformal (CQR): kuantil model dilebarkan/dipersempit `offset`."""
    lo, mid, hi = raw_quantiles(models, X)
    new_lo = np.minimum(lo - offset, mid)
    new_hi = np.maximum(hi + offset, mid)
    new_lo = np.clip(new_lo, lower, upper)
    new_hi = np.clip(new_hi, lower, upper)
    mid = np.clip(mid, new_lo, new_hi)
    return new_lo, mid, new_hi


def confidence_features(soh_lo, soh_mid, soh_hi, thresholds=None):
    """Fitur untuk skor keyakinan: lebar interval & jarak ke batas grade terdekat."""
    t = thresholds or DEFAULT_THRESHOLDS
    bounds = np.array(sorted(t.values()))
    soh_lo, soh_mid, soh_hi = (np.asarray(a, dtype=float) for a in (soh_lo, soh_mid, soh_hi))
    width = soh_hi - soh_lo
    dist = np.min(np.abs(soh_mid[:, None] - bounds[None, :]), axis=1)
    crosses = ((soh_lo[:, None] < bounds[None, :]) & (soh_hi[:, None] >= bounds[None, :])).any(axis=1)
    return np.column_stack([width, dist, dist / (width + 1e-6), crosses.astype(float)])


def confidence_score(conf_model, feats):
    """Perkiraan probabilitas bahwa grade yang dikeluarkan benar (0-1)."""
    if isinstance(conf_model, float):  # fallback bila kalibrasi tidak punya 2 kelas
        return np.full(len(feats), conf_model)
    return conf_model.predict_proba(feats)[:, 1]


# --------------------------------------------------------------------------- #
# Predictor untuk web/API
# --------------------------------------------------------------------------- #
class BatteryHealthPredictor:
    PASSPORT_VERSION = "1.0"

    def __init__(self, bundle_path="artifacts/battery_health_bundle.joblib"):
        self.bundle = joblib.load(Path(bundle_path))
        b = self.bundle
        self.meta = b["meta"]
        self.thresholds = b["grade_thresholds"]
        self.num_features = b["features"]["numeric"]
        self.cat_features = b["features"]["categorical"]
        self.all_features = self.num_features + self.cat_features
        trained_with = self.meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Model dilatih dgn scikit-learn {trained_with}, sekarang {sklearn.__version__}. "
                "Samakan versinya di server (lihat requirements di model card)."
            )

    # -- DataFrame in -> DataFrame out ------------------------------------- #
    def _prepare(self, df):
        df = df.copy()
        for c in self.all_features:
            if c not in df.columns:
                df[c] = np.nan
        for c in self.num_features:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in self.cat_features:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        return df[self.all_features]

    def predict_dataframe(self, df):
        b = self.bundle
        X_raw = self._prepare(df)
        X = b["preprocessor"].transform(X_raw)

        s_lo, s_mid, s_hi = predict_interval(b["soh_models"], b["soh_offset"], X, 0.0, 100.0)
        r_lo, r_mid, r_hi = predict_interval(b["rul_models"], b["rul_offset"], X, 0.0, None)
        feats = confidence_features(s_lo, s_mid, s_hi, self.thresholds)
        conf = confidence_score(b["conf_model"], feats)
        completeness = X_raw.notna().mean(axis=1).to_numpy()
        # Model dilatih dgn data yg hampir lengkap -> bila input banyak kosong,
        # keyakinan dibatasi oleh kelengkapan input (konservatif).
        conf = np.where(completeness < MIN_COMPLETENESS, np.minimum(conf, completeness), conf)

        out = pd.DataFrame(
            {
                "grade": soh_to_grade(s_mid, self.thresholds),
                "grade_worst_case": soh_to_grade(s_lo, self.thresholds),
                "grade_best_case": soh_to_grade(s_hi, self.thresholds),
                "soh_lower": s_lo, "soh_mid": s_mid, "soh_upper": s_hi,
                "rul_lower": r_lo, "rul_mid": r_mid, "rul_upper": r_hi,
                "confidence": conf,
                "input_completeness": completeness,
            },
            index=df.index,
        )
        return out

    # -- dict(s) in -> passport(s) out (JSON-ready) ------------------------- #
    def predict(self, records):
        """records: satu dict atau list of dict. Return passport (dict) atau list."""
        single = isinstance(records, dict)
        recs = [records] if single else list(records)
        df = pd.DataFrame(recs)
        res = self.predict_dataframe(df)
        passports = [self._passport(r, recs[i]) for i, (_, r) in enumerate(res.iterrows())]
        return passports[0] if single else passports

    def _passport(self, r, rec):
        warnings_ = []
        if r["input_completeness"] < MIN_COMPLETENESS:
            warnings_.append("Data input kurang lengkap (<80% fitur terisi); hasil kurang andal dan keyakinan dibatasi.")
        if r["grade_worst_case"] != r["grade_best_case"]:
            warnings_.append("Interval SOH melewati batas grade; grade bisa berbeda dari yang ditampilkan.")
        m = self.meta
        return {
            "passport_version": self.PASSPORT_VERSION,
            "battery_id": rec.get("battery_id"),
            "assessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "grade": str(r["grade"]),
            "grade_range": {"worst_case": str(r["grade_worst_case"]), "best_case": str(r["grade_best_case"])},
            "state_of_health_pct": {
                "lower": round(float(r["soh_lower"]), 1),
                "mid": round(float(r["soh_mid"]), 1),
                "upper": round(float(r["soh_upper"]), 1),
            },
            "remaining_life_cycles": {
                "lower": int(round(r["rul_lower"])),
                "mid": int(round(r["rul_mid"])),
                "upper": int(round(r["rul_upper"])),
            },
            "interval_level": m.get("interval_level"),
            "confidence": round(float(r["confidence"]), 3),
            "confidence_meaning": "Perkiraan probabilitas grade benar; dibatasi kelengkapan input bila <80% fitur terisi.",
            "input_completeness": round(float(r["input_completeness"]), 2),
            "warnings": warnings_,
            "model": {
                "version": m.get("model_version"),
                "trained_at": m.get("trained_at"),
                "dataset": m.get("dataset"),
            },
        }
