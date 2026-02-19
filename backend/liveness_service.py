from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
import time
import traceback
import numpy as np
from numpy.linalg import norm

# import your liveness module (adjust name/path if different)
import main_liveness as liveness

# ---------- basic logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("liveness_service")

# ---------- FastAPI app ----------
app = FastAPI(title="Liveness Service")

# Allow CORS (useful for local web tests). Adjust origins as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change "*" to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Fused verification config
# -----------------------
ALPHA = 0.75           # weight for cosine in fused score
FUSED_THRESHOLD = 0.65 # default threshold for match
EPS = 1e-8             # small value to prevent division by zero

# -----------------------
# Helper functions
# -----------------------
def stabilize_and_norm(emb: np.ndarray) -> np.ndarray:
    """
    Stabilize embedding: handle NaN/inf, zero-center, normalize variance, and L2 normalize.
    """
    emb = np.nan_to_num(emb)
    emb = emb - np.mean(emb)
    std = np.std(emb) + 1e-6
    emb = emb / std
    n = np.linalg.norm(emb) + EPS
    return emb / n

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((norm(a) * norm(b)) + EPS))

def euclid_to_similarity(d: float) -> float:
    """
    Map Euclidean distance -> similarity [0,1]
    """
    return float(1.0 / (1.0 + d))

def fused_score(emb_a: np.ndarray, emb_b: np.ndarray, alpha: float = ALPHA) -> dict:
    """
    Compute fused similarity score combining cosine and Euclidean distance.
    Convert all values to native Python float to make them JSON serializable.
    """
    cos = cosine_similarity(emb_a, emb_b)
    d = float(np.linalg.norm(emb_a - emb_b))
    e_sim = euclid_to_similarity(d)
    fused = float(alpha * cos + (1.0 - alpha) * e_sim)
    return {
        "cosine": float(cos),
        "euclid": d,
        "euclid_similarity": e_sim,
        "fused": fused
    }

# -----------------------
# Existing endpoint: start-liveness
# -----------------------
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
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "traceback": tb})

    try:
        if result is None:
            return JSONResponse(status_code=500, content={"success": False, "error": "No result returned from liveness."})

        meta = result.get("meta") if isinstance(result, dict) else None
        if isinstance(meta, dict):
            emb = meta.get("embedding")
            if hasattr(emb, "tolist"):
                try:
                    meta["embedding"] = emb.tolist()
                except Exception:
                    meta.pop("embedding", None)

        return JSONResponse(status_code=200, content={"success": True, "result": result})
    except Exception as e:
        log.exception("Failed to normalise result for JSON:")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# -----------------------
# New endpoint: verify embeddings
# -----------------------
@app.post("/verify")
def verify_embeddings(payload: dict):
    """
    Verify two embeddings using fused score.
    Returns a clean JSON including status for easy destructuring.
    """
    try:
        emb1 = np.array(payload.get("embedding1"), dtype=np.float32)
        emb2 = np.array(payload.get("embedding2"), dtype=np.float32)
        threshold = float(payload.get("threshold", FUSED_THRESHOLD))

        if emb1.shape != emb2.shape:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Embeddings must have the same shape."}
            )

        emb1 = stabilize_and_norm(emb1)
        emb2 = stabilize_and_norm(emb2)

        metrics = fused_score(emb1, emb2, alpha=ALPHA)
        match = metrics['fused'] >= threshold
        status = "same_person" if match else "different_person"

        response = {
            "success": True,
            "match": match,
            "status": status,
            "metrics": metrics
        }
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        log.exception("Error verifying embeddings:")
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "traceback": tb})

# -----------------------
# Health check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("LIVENESS_PORT", 7700))
    host = os.environ.get("LIVENESS_HOST", "0.0.0.0")
    log.info("Starting Liveness service on %s:%d", host, port)
    uvicorn.run("liveness_service:app", host=host, port=port, reload=False, log_level="info")
