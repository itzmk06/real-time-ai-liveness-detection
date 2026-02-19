import numpy as np
from numpy.linalg import norm
import os

# -----------------------
# Config (tune this)
# -----------------------
embedding1_path = r"E:\Liveness-Detection-main\Liveness-Detection-main\Model (Using Dataset 600 Images)\backend\embeddings\embed_20251123_061827.npy"
embedding2_path = r"E:\Liveness-Detection-main\Liveness-Detection-main\Model (Using Dataset 600 Images)\backend\embeddings\embed_20251122_190842.npy"

# Fusion params (start with these; calibrate later)
ALPHA = 0.75   # weight for cosine in fused score
FUSED_THRESHOLD = 0.65  # above this => same person (tune per model/data)

EPS = 1e-8

# -----------------------
# Helpers
# -----------------------
def load_emb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    e = np.load(path).astype(np.float32)
    return e

def stabilize_and_norm(emb):
    emb = np.nan_to_num(emb)           # drop NaN/inf
    emb = emb - np.mean(emb)          # zero center
    std = np.std(emb) + 1e-6
    emb = emb / std                    # normalize variance
    n = np.linalg.norm(emb) + EPS
    return emb / n

def cosine_similarity(a, b):
    return float(np.dot(a, b) / ((norm(a) * norm(b)) + EPS))

def euclid_to_similarity(d):
    # Map Euclidean distance -> [0,1] similarity.
    # 1/(1+d) works well: distance 0 -> 1, large d -> 0
    return 1.0 / (1.0 + d)

def fused_score(emb_a, emb_b, alpha=ALPHA):
    # requires emb_a, emb_b normalized
    cos = cosine_similarity(emb_a, emb_b)
    d = np.linalg.norm(emb_a - emb_b)
    e_sim = euclid_to_similarity(d)
    fused = alpha * cos + (1.0 - alpha) * e_sim
    return {
        "cosine": cos,
        "euclid": d,
        "euclid_similarity": e_sim,
        "fused": fused
    }

# -----------------------
# Main
# -----------------------
a = load_emb(embedding1_path)
b = load_emb(embedding2_path)

a = stabilize_and_norm(a)
b = stabilize_and_norm(b)

metrics = fused_score(a, b, alpha=ALPHA)
print(f"Cosine Similarity: {metrics['cosine']:.4f}")
print(f"Euclidean Distance: {metrics['euclid']:.4f}")
print(f"Euclid->Similarity: {metrics['euclid_similarity']:.4f}")
print(f"Fused score (alpha={ALPHA}): {metrics['fused']:.4f}")

if metrics['fused'] >= FUSED_THRESHOLD:
    print("✅ The embeddings match (fused). Likely the same person.")
else:
    print("❌ The embeddings do NOT match (fused). Likely a different person.")
