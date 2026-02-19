# liveness_service.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
import time
import traceback
import numpy as np
from numpy.linalg import norm

# import your liveness module
import main_liveness as liveness

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("liveness_service")

# ---------- FastAPI ----------
app = FastAPI(title="Liveness Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Config for Final Logic 4 ----------------------
ALPHA = 0.75              # weight for cosine in fused score
BASE_THRESHOLD = 0.65     # base threshold for live/spoof
EPS = 1e-8
OUTLIER_STD_CLIP = 3.0    # clip extreme embedding values

# ---------------------- Helper Functions ----------------------
def stabilize_and_norm(emb):
    emb = np.nan_to_num(emb)
    mean, std = np.mean(emb), np.std(emb) + EPS
    emb = (emb - mean) / std
    emb = np.clip(emb, -OUTLIER_STD_CLIP, OUTLIER_STD_CLIP)
    n = np.linalg.norm(emb) + EPS
    return emb / n

def cosine_similarity(a, b):
    return float(np.dot(a, b) / ((norm(a) * norm(b)) + EPS))

def euclid_to_similarity(d):
    return 1.0 / (1.0 + d)

def mahalanobis_similarity(a, b):
    cov = np.cov(np.stack([a, b]), rowvar=False) + np.eye(len(a)) * EPS
    diff = a - b
    try:
        inv_cov = np.linalg.inv(cov)
        m_dist = np.sqrt(diff.T @ inv_cov @ diff)
        return 1.0 / (1.0 + m_dist)
    except np.linalg.LinAlgError:
        return euclid_to_similarity(np.linalg.norm(diff))

def fused_score(emb_a, emb_b, alpha=ALPHA):
    cos = cosine_similarity(emb_a, emb_b)
    euclid_d = np.linalg.norm(emb_a - emb_b)
    e_sim = euclid_to_similarity(euclid_d)
    maha_sim = mahalanobis_similarity(emb_a, emb_b)
    fused = alpha * cos + 0.25 * e_sim + 0.25 * maha_sim
    return {
        "cosine": cos,
        "euclidean_distance": euclid_d,
        "euclid_similarity": e_sim,
        "mahalanobis_similarity": maha_sim,
        "fused": fused
    }

def adaptive_threshold(emb_a, emb_b):
    mag = (np.linalg.norm(emb_a) + np.linalg.norm(emb_b)) / 2
    return BASE_THRESHOLD + (0.1 if mag < 0.8 else 0.0)

def verify_embeddings_final(embedding1: list, embedding2: list):
    try:
        emb1 = stabilize_and_norm(np.array(embedding1, dtype=np.float32))
        emb2 = stabilize_and_norm(np.array(embedding2, dtype=np.float32))
        metrics = fused_score(emb1, emb2, alpha=ALPHA)
        threshold = adaptive_threshold(emb1, emb2)
        verdict = "live" if metrics["fused"] >= threshold else "spoof"
        confidence = min(max(metrics["fused"], 0.0), 1.0)
        return {
            "success": True,
            "verdict": verdict,
            "confidence": round(confidence, 6),
            "metrics": {
                "cosine_similarity": round(metrics["cosine"], 6),
                "euclidean_distance": round(metrics["euclidean_distance"], 6),
                "euclid_similarity": round(metrics["euclid_similarity"], 6),
                "mahalanobis_similarity": round(metrics["mahalanobis_similarity"], 6),
                "fused_score": round(metrics["fused"], 6),
                "threshold": round(threshold, 6)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# ---------------------- Endpoint: Start Liveness ----------------------
@app.post("/start-liveness")
def start_liveness(model_path: str = None, camera_index: int = 0, timeout_seconds: int = 60, show_window: bool = True):
    if model_path is None:
        try:
            model_path = liveness.DEFAULT_CNN_MODEL
        except Exception:
            model_path = None

    log.info("Incoming request: start-liveness model=%s camera=%s timeout=%s show_window=%s",
             model_path, camera_index, timeout_seconds, show_window)

    try:
        result = liveness.run_liveness_once(
            model_path=model_path,
            camera_index=camera_index,
            timeout_seconds=timeout_seconds,
            show_window=show_window
        )
    except Exception as e:
        log.exception("Unhandled exception running liveness:")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "traceback": traceback.format_exc()})

    # Normalize embeddings for JSON
    meta = result.get("meta") if isinstance(result, dict) else None
    if isinstance(meta, dict):
        emb = meta.get("embedding")
        if hasattr(emb, "tolist"):
            try:
                meta["embedding"] = emb.tolist()
            except Exception:
                meta.pop("embedding", None)
    return JSONResponse(status_code=200, content={"success": True, "result": result})

# ---------------------- Endpoint: Verify Embeddings ----------------------
@app.post("/verify")
def verify(embedding1: list, embedding2: list):
    """
    Ultra-robust verification endpoint using Final Logic 4.
    Returns simple verdict: 'live' or 'spoof' with detailed metrics.
    """
    result = verify_embeddings_final(embedding1, embedding2)
    return JSONResponse(status_code=200 if result["success"] else 500, content=result)

# ---------------------- Health Check ----------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}

# ---------------------- Run Server ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("LIVENESS_PORT", 7700))
    host = os.environ.get("LIVENESS_HOST", "0.0.0.0")
    log.info("Starting Liveness service on %s:%d", host, port)
    uvicorn.run("liveness_service:app", host=host, port=port, reload=False, log_level="info")
