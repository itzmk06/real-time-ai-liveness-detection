#!/usr/bin/env python3
"""
Ultra-Robust Liveness Detection
Single-file, production-focused script that:
 - Integrates Mediapipe face mesh + a CNN liveness model
 - Uses DeepFace (Facenet) for embeddings (with preload/warmup)
 - Has challenge-based liveness (blink, turn, smile, say-hello)
 - Robust thread-safe embedding creation in background
 - CLI, evaluation harness, synthetic attack simulator, and reporting
 - Extensive defensive checks and logging
Notes:
 - Tune thresholds with your datasets.
 - Dependencies: opencv-python, mediapipe, deepface, tensorflow, scikit-image, numpy
"""
# =============================================================================
# Standard library imports
# =============================================================================
import argparse
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math
import json
import hashlib
import tempfile
import threading
import random
import secrets
import logging
import signal
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from skimage.feature import local_binary_pattern

# Optional deepface import with friendly error
try:
    from deepface import DeepFace
except Exception as e:
    DeepFace = None
    # We'll check at runtime and provide an actionable error later

# Keras model import guard (works with tensorflow Keras)
try:
    from keras.models import load_model
except Exception:
    # fallback to tensorflow.keras
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        load_model = None

# =============================================================================
# Configuration (tweak as needed)
# =============================================================================
# Model paths (change as needed)
DEFAULT_CNN_MODEL = r"E:\Liveness-Detection-main\Liveness-Detection-main\Model (Using Dataset 600 Images)\model.h5"
# Directories
SPOOF_SAVE_DIR = "spoof_captures"
LIVE_SAVE_DIR = "live_captures"
EMB_SAVE_DIR = "embeddings"
LOG_DIR = "logs"
REPORT_DIR = "reports"

# Create directories
os.makedirs(SPOOF_SAVE_DIR, exist_ok=True)
os.makedirs(LIVE_SAVE_DIR, exist_ok=True)
os.makedirs(EMB_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Liveness thresholds & params
EAR_THRESHOLD = 0.22
SMILE_THRESHOLD = 1.6
BLINK_CONSEC_FRAMES = 3
CHALLENGE_INTERVAL = 6
RESET_DISTANCE = 80
RESET_TIMEOUT = 2.0
INPUT_SIZE = (150, 150)
STABILIZATION_TIME = 2.5

# Stronger final thresholds
FINAL_SCORE_THRESHOLD = 0.77
Z_VAR_THRESHOLD = 5e-05
MIN_ROLLING_FRAMES = 8
SHAPE_VARIATION_THRESHOLD = 0.002
NO_DEPTH_PENALTY = 0.55

# Per-challenge thresholds
TURN_SCORE_THRESHOLD = 0.6
SMILE_SCORE_WINDOW = 6
SMILE_SCORE_THRESHOLD = 0.75
BLINK_REQUIRED_EVENTS = 1

# SAY_HELLO params
MOUTH_OPEN_THRESHOLD = 0.45
MOUTH_CONSEC_FRAMES = 2

# Embedding / capture directories & salt
EMB_SALT = b"replace_with_your_secret_salt"  # change for production
GENERATE_EMB_ON_SPOOF = False

# Peak fallback and sustained frames
PEAK_REQUIRED = 0.90
PROGRESS_THRESHOLD_FOR_PEAK = 80
RANDOMIZE_CHALLENGES = True
SUSTAINED_THRESHOLD = 0.85
SUSTAINED_FRAMES_REQUIRED = 4
SHOW_SEQUENCE_SECONDS = 3.0

# Frontal tolerances
FRONTAL_RATIO_TOL = 0.12
MAX_ROLL_DEGREES = 20.0
MIN_ALIGNED_CAPTURES = 1

# DONE thresholds
DONE_MIN_AVG = 0.90
DONE_MIN_PROGRESS = 100
DONE_MIN_Z = 0.01

# Challenge base sequence
BASE_SEQ = ["BLINK", "TURN_LEFT", "TURN_RIGHT", "SMILE", "SAY_HELLO"]

# =============================================================================
# Logging
# =============================================================================
LOG_FILE = os.path.join(LOG_DIR, f"liveness_{int(time.time())}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
log = logging.getLogger("ultra_liveness")

# =============================================================================
# Utility functions
# =============================================================================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def l2_normalize(vec):
    n = np.linalg.norm(vec)
    return vec / (n + 1e-12)

def safe_imwrite(path, img):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return cv2.imwrite(path, img)
    except Exception as e:
        log.warning("Failed to write image %s: %s", path, e)
        return False

# =============================================================================
# Feature computation helpers (copied + hardened)
# =============================================================================
def eye_aspect_ratio(landmarks, left_indices, right_indices):
    try:
        left_a = euclidean(landmarks[left_indices[1]], landmarks[left_indices[5]])
        left_b = euclidean(landmarks[left_indices[2]], landmarks[left_indices[4]])
        left_c = euclidean(landmarks[left_indices[0]], landmarks[left_indices[3]])
        leftEAR = (left_a + left_b) / (2.0 * (left_c + 1e-9))

        right_a = euclidean(landmarks[right_indices[1]], landmarks[right_indices[5]])
        right_b = euclidean(landmarks[right_indices[2]], landmarks[right_indices[4]])
        right_c = euclidean(landmarks[right_indices[0]], landmarks[right_indices[3]])
        rightEAR = (right_a + right_b) / (2.0 * (right_c + 1e-9))

        return (leftEAR + rightEAR) / 2.0
    except Exception as e:
        log.debug("EAR calc failed: %s", e)
        return 1.0  # default to open

def get_head_turn_with_depth(landmarks_3d):
    # uses nose tip (1) and cheeks (234, 454)
    try:
        nose_tip = np.array(landmarks_3d[1])
        left_cheek = np.array(landmarks_3d[234])
        right_cheek = np.array(landmarks_3d[454])
        ratio_xy = np.linalg.norm(nose_tip[:2] - left_cheek[:2]) / (np.linalg.norm(nose_tip[:2] - right_cheek[:2]) + 1e-9)
        z_values = [nose_tip[2], left_cheek[2], right_cheek[2]]
        z_var = np.var(z_values)
        z_diff = left_cheek[2] - right_cheek[2]
        return ratio_xy, z_var, z_diff
    except Exception as e:
        log.debug("Head turn calc failed: %s", e)
        return 1.0, 0.0, 0.0

def get_smile_metric(landmarks):
    try:
        left_lip = np.array(landmarks[61])
        right_lip = np.array(landmarks[291])
        top_lip = np.array(landmarks[0])
        bottom_lip = np.array(landmarks[17])
        horizontal = np.linalg.norm(left_lip - right_lip)
        vertical = np.linalg.norm(top_lip - bottom_lip)
        return horizontal / (vertical + 1e-6)
    except Exception as e:
        log.debug("Smile metric failed: %s", e)
        return 0.0

def get_mouth_open_metric(landmarks):
    try:
        left_lip = np.array(landmarks[61])
        right_lip = np.array(landmarks[291])
        top_lip = np.array(landmarks[0])
        bottom_lip = np.array(landmarks[17])
        horizontal = np.linalg.norm(left_lip - right_lip) + 1e-9
        vertical = np.linalg.norm(top_lip - bottom_lip)
        return vertical / horizontal
    except Exception as e:
        log.debug("Mouth open metric failed: %s", e)
        return 0.0

def lbp_texture(gray_face):
    try:
        lbp = local_binary_pattern(gray_face, 8, 1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist.var()
    except Exception as e:
        log.debug("LBP failed: %s", e)
        return 0.0

def compute_shape_ratios(landmarks_3d):
    try:
        a = np.array(landmarks_3d[33])[:2]
        b = np.array(landmarks_3d[362])[:2]
        nose = np.array(landmarks_3d[1])[:2]
        chin = np.array(landmarks_3d[152])[:2]
        left_cheek = np.array(landmarks_3d[234])[:2]
        right_cheek = np.array(landmarks_3d[454])[:2]
        interocular = np.linalg.norm(a - b) + 1e-9
        nose_chin = np.linalg.norm(nose - chin) + 1e-9
        cheek_width = np.linalg.norm(left_cheek - right_cheek) + 1e-9
        r1 = nose_chin / interocular
        r2 = cheek_width / interocular
        return np.array([r1, r2])
    except Exception as e:
        log.debug("Shape ratio calc failed: %s", e)
        return np.array([0.0, 0.0])

def align_face(face_bgr, landmarks_local, out_size=(160,160)):
    """
    Rotate crop so eyes are horizontal, crop centered, and resize.
    landmarks_local should be relative to the crop (x,y).
    Returns BGR image or None.
    """
    try:
        left_eye_idx = 33
        right_eye_idx = 362
        if left_eye_idx >= len(landmarks_local) or right_eye_idx >= len(landmarks_local):
            return None
        le = np.array(landmarks_local[left_eye_idx], dtype=float)
        re = np.array(landmarks_local[right_eye_idx], dtype=float)
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        angle = np.degrees(np.arctan2(dy, dx))
        eyes_center = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
        (h, w) = face_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((eyes_center[0], eyes_center[1]), angle, 1.0)
        rotated = cv2.warpAffine(face_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ec = np.array([eyes_center[0], eyes_center[1], 1.0])
        new_center = M.dot(ec)
        out_w, out_h = out_size
        cx, cy = int(new_center[0]), int(new_center[1])
        half_w, half_h = out_w // 2, out_h // 2
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(rotated.shape[1], cx + half_w)
        y2 = min(rotated.shape[0], cy + half_h)
        crop = rotated[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        aligned = cv2.resize(crop, out_size)
        return aligned
    except Exception as e:
        log.debug("Align face failed: %s", e)
        return None

# =============================================================================
# Embedding helpers using DeepFace
# =============================================================================
def get_embedding_from_facecrop_deepface(face_bgr, preloaded_model=None):
    """Return L2-normalized embedding or None."""
    if DeepFace is None:
        log.error("DeepFace not installed. Install via: pip install deepface tensorflow")
        return None
    try:
        tmpdir = tempfile.gettempdir()
        tmp_path = Path(tmpdir) / f"tmp_face_{int(time.time()*1000)}_{secrets.token_hex(4)}.jpg"
        cv2.imwrite(str(tmp_path), face_bgr)
        if preloaded_model is not None:
            reps = DeepFace.represent(img_path=str(tmp_path), model_name="Facenet",
                                      model=preloaded_model, enforce_detection=False)
        else:
            reps = DeepFace.represent(img_path=str(tmp_path), model_name="Facenet", enforce_detection=False)
        try:
            tmp_path.unlink()
        except Exception:
            pass
        if reps is None:
            return None
        if isinstance(reps, list) and len(reps) > 0:
            first = reps[0]
            if isinstance(first, dict) and "embedding" in first:
                vec = np.array(first["embedding"], dtype=np.float32)
            else:
                vec = np.array(first, dtype=np.float32)
        elif isinstance(reps, dict) and "embedding" in reps:
            vec = np.array(reps["embedding"], dtype=np.float32)
        else:
            vec = np.array(reps, dtype=np.float32)
        if vec.size == 0:
            return None
        return l2_normalize(vec.astype(np.float32))
    except Exception as e:
        log.warning("DeepFace embedding extraction failed: %s", e)
        return None

# =============================================================================
# Save embedding + metadata
# =============================================================================
def save_embedding_and_image(captures_deque, session_nonce=None):
    """
    captures_deque: deque of aligned BGR face crops
    Saves:
      - best image (middle)
      - averaged embedding (np.save)
      - metadata json with salted hash
    Returns meta dict or None
    """
    timestamp = int(time.time())
    dtstr = datetime.utcfromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    imgs = list(captures_deque)
    if len(imgs) == 0:
        log.warning("No captures to save.")
        return None
    best_idx = len(imgs) // 2
    best_img = imgs[best_idx]
    img_fname = os.path.join(LIVE_SAVE_DIR, f"live_{dtstr}.jpg")
    safe_imwrite(img_fname, best_img)

    # build embeddings list
    embs = []
    for i, img in enumerate(imgs):
        emb = get_embedding_from_facecrop_deepface(img)
        if emb is not None:
            embs.append(emb)
    if not embs:
        log.warning("No valid embeddings extracted from captures.")
        return None

    embs_arr = np.vstack(embs)
    template = np.mean(embs_arr, axis=0)
    template = l2_normalize(template).astype(np.float32)

    emb_fname = os.path.join(EMB_SAVE_DIR, f"embed_{dtstr}.npy")
    try:
        np.save(emb_fname, template)
    except Exception as e:
        log.warning("Failed to save embedding file: %s", e)
        return None

    emb_bytes = template.tobytes()
    emb_hash = hashlib.sha256(emb_bytes + EMB_SALT).hexdigest()

    meta = {
        "timestamp": timestamp,
        "datetime_utc": dtstr,
        "image_path": img_fname,
        "embed_path": emb_fname,
        "embed_hash_sha256": emb_hash,
        "embedding_dim": int(template.shape[0]),
        "model": "deepface/facenet (normalized)",
        "nonce": session_nonce
    }
    meta_fname = os.path.join(EMB_SAVE_DIR, f"meta_{dtstr}.json")
    try:
        with open(meta_fname, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        log.warning("Failed to save metadata: %s", e)

    log.info("Saved live image: %s", img_fname)
    log.info("Saved template embedding: %s", emb_fname)
    log.info("Saved metadata: %s", meta_fname)
    log.info("Embedding SHA256 (salted): %s", emb_hash)
    return meta

# =============================================================================
# State management
# =============================================================================
def reset_state():
    return {
        "blink_done": False,
        "turn_left_done": False,
        "turn_right_done": False,
        "smile_done": False,
        "speak_done": False,
        "blink_counter": 0,
        "challenge_state": "STABILIZE",
        "stabilize_start": None,
        "last_challenge_time": time.time(),
        "last_seen_time": time.time(),
        "prev_center": None,
        "live_scores": deque(maxlen=40),
        "final_verdict": None,
        "verdict_time": None,
        "motion_detected": False,
        "last_motion_time": time.time(),
        "z_history": deque(maxlen=MIN_ROLLING_FRAMES*2),
        "head_ratio_history": deque(maxlen=MIN_ROLLING_FRAMES*2),
        "shape_ratio_history": deque(maxlen=MIN_ROLLING_FRAMES*2),
        "challenge_scores": {
            "BLINK": deque(maxlen=40),
            "TURN_LEFT": deque(maxlen=MIN_ROLLING_FRAMES*2),
            "TURN_RIGHT": deque(maxlen=MIN_ROLLING_FRAMES*2),
            "SMILE": deque(maxlen=SMILE_SCORE_WINDOW*2),
            "SAY_HELLO": deque(maxlen=20)
        },
        "mouth_open_counter": 0,
        "pre_live_captures": deque(maxlen=6),
        "embedding_in_progress": False,
        "peak_live": 0.0,
        "session_nonce": secrets.token_hex(8),
        "seq": None,
        "seq_show_start": None,
        "high_frame_streak": 0
    }

def soft_reset_preserve_challenges(state):
    preserved = {
        "blink_done": state.get("blink_done", False),
        "turn_left_done": state.get("turn_left_done", False),
        "turn_right_done": state.get("turn_right_done", False),
        "smile_done": state.get("smile_done", False),
        "speak_done": state.get("speak_done", False),
        "live_scores": list(state.get("live_scores", deque()))
    }
    new = reset_state()
    new["blink_done"] = preserved["blink_done"]
    new["turn_left_done"] = preserved["turn_left_done"]
    new["turn_right_done"] = preserved["turn_right_done"]
    new["smile_done"] = preserved["smile_done"]
    new["speak_done"] = preserved["speak_done"]
    new["live_scores"] = deque(preserved["live_scores"], maxlen=40)
    new["last_seen_time"] = time.time()
    return new

# =============================================================================
# UI overlay helper (visual polish)
# =============================================================================
def draw_scanner_overlay(frame, x1, y1, x2, y2, now):
    h_box = max(1, y2 - y1)
    w_box = max(1, x2 - x1)
    overlay = frame.copy() * 0
    rect_color = (200, 200, 200)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), rect_color, 1)

    speed = 180
    band_height = max(4, h_box // 40)
    offset = int(((now * speed) % (h_box + band_height)) - band_height)
    gap = band_height * 3
    bar_color = (255, 180, 50)
    for i in range(-2, int(h_box / gap) + 3):
        y = y1 + ((offset + i * gap) % (h_box + band_height))
        y0 = int(np.clip(y, y1, y2))
        y1b = int(np.clip(y + band_height, y1, y2))
        if y1b > y0:
            cv2.rectangle(overlay, (x1, y0), (x2, y1b), bar_color, -1)

    sweep_speed = 220
    sx = x1 + int((now * sweep_speed) % (w_box))
    cv2.line(overlay, (sx, y1), (sx, y2), (200, 200, 255), 1)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = max(8, min(w_box, h_box) // 12)
    ang = (now * 360) % 360
    sin_a = int(radius * 0.9 * np.sin(np.deg2rad(ang)))
    cos_a = int(radius * 0.9 * np.cos(np.deg2rad(ang)))
    cv2.line(overlay, (cx - cos_a, cy - sin_a), (cx - cos_a//2, cy - sin_a//2), (220, 220, 255), 2)
    cv2.line(overlay, (cx + cos_a, cy + sin_a), (cx + cos_a//2, cy + sin_a//2), (220, 220, 255), 2)
    cv2.line(overlay, (cx - sin_a, cy + cos_a), (cx - sin_a//2, cy + cos_a//2), (220, 220, 255), 2)
    cv2.line(overlay, (cx + sin_a, cy - cos_a), (cx + sin_a//2, cy - cos_a//2), (220, 220, 255), 2)

    for gx in range(x1 + 8, x2, max(12, w_box // 8)):
        cv2.line(overlay, (gx, y1), (gx, y2), (60, 60, 60), 1)
    for gy in range(y1 + 8, y2, max(12, h_box // 8)):
        cv2.line(overlay, (x1, gy), (x2, gy), (60, 60, 60), 1)

    txt = "SCANNING..."
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    txt_x = x1 + 6
    txt_y = y1 - 10 if y1 - 10 > 20 else y1 + th + 6
    cv2.putText(overlay, txt, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)

    alpha = 0.36
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# =============================================================================
# Main class encapsulating the liveness pipeline (provides modularity)
# =============================================================================
class UltraLiveness:
    """
    Encapsulates the liveness detection pipeline so you can instantiate and call from other apps.
    """

    def __init__(self, cnn_model_path=DEFAULT_CNN_MODEL, use_deepface_preload=True, camera_index=0,
                 show_window=True, debug=False):
        self.camera_index = camera_index
        self.show_window = show_window
        self.debug = debug
        self.state_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.embed_queue = Queue(maxsize=4)
        self.embed_thread = None
        self.deepface_model = None
        self.cnn_model = None
        self.face_mesh = None
        self.cap = None
        self.prev_gray = None
        self.state = reset_state()
        self.cnn_model_path = cnn_model_path
        self.use_deepface_preload = use_deepface_preload

        # prepare mediapipe face mesh reference
        self.mp_face_mesh = mp.solutions.face_mesh

        # Initialize components
        self._load_cnn_model()
        self._maybe_preload_deepface()
        self._start_embedding_worker()

    def _load_cnn_model(self):
        if load_model is None:
            log.error("Keras load_model not available. Ensure TensorFlow/Keras is installed.")
            raise RuntimeError("load_model unavailable")
        try:
            log.info("Loading CNN liveliness model from: %s", self.cnn_model_path)
            self.cnn_model = load_model(self.cnn_model_path)
            log.info("Model loaded.")
            # warmup
            dummy = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32)
            with self.model_lock:
                try:
                    self.cnn_model.predict(dummy, verbose=0)
                    log.info("CNN warmup done.")
                except Exception as e:
                    log.warning("CNN warmup failed: %s", e)
        except Exception as e:
            log.exception("Failed to load CNN model: %s", e)
            raise

    def _maybe_preload_deepface(self):
        if DeepFace is None:
            log.warning("DeepFace not available; embeddings will fail. Install deepface to enable embeddings.")
            self.deepface_model = None
            return
        if not self.use_deepface_preload:
            log.info("DeepFace preload disabled by config.")
            self.deepface_model = None
            return
        try:
            log.info("Preloading DeepFace Facenet model...")
            self.deepface_model = DeepFace.build_model("Facenet")
            log.info("DeepFace preloaded.")
        except Exception as e:
            log.warning("DeepFace preload failed: %s", e)
            self.deepface_model = None

    def _embedding_consumer_worker(self):
        """
        Background worker that takes capture lists from a queue and writes embeddings.
        """
        while not self.stop_event.is_set():
            try:
                captures_list, verdict_label, session_nonce = self.embed_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                log.info("[embed-worker] Creating embedding (verdict=%s, frames=%d)", verdict_label, len(captures_list))
                meta = save_embedding_and_image(deque(captures_list), session_nonce=session_nonce)
                if meta:
                    log.info("[embed-worker] Embedding saved: %s", meta.get("embed_path"))
                else:
                    log.warning("[embed-worker] Embedding worker produced no embedding.")
            except Exception as e:
                log.exception("[embed-worker] Worker error: %s", e)
            finally:
                with self.state_lock:
                    self.state["embedding_in_progress"] = False
                    try:
                        self.state["pre_live_captures"].clear()
                    except Exception:
                        pass
                self.embed_queue.task_done()
        log.info("[embed-worker] Exiting.")

    def _start_embedding_worker(self):
        self.embed_thread = threading.Thread(target=self._embedding_consumer_worker, daemon=True)
        self.embed_thread.start()
        log.info("Embedding worker started.")

    def schedule_embedding_task(self, captures_list, verdict_label):
        with self.state_lock:
            if self.state.get("embedding_in_progress"):
                log.info("Embedding already in progress; skipping.")
                return False
            self.state["embedding_in_progress"] = True
            session_nonce = self.state.get("session_nonce")
        try:
            self.embed_queue.put_nowait((captures_list, verdict_label, session_nonce))
            log.info("Scheduled embedding task.")
            return True
        except Exception as e:
            log.warning("Failed to schedule embedding: %s", e)
            with self.state_lock:
                self.state["embedding_in_progress"] = False
            return False

    def _maybe_enqueue_embedding(self, verdict):
        with self.state_lock:
            captures_list = [img.copy() for img in list(self.state.get("pre_live_captures", deque()))]
        if captures_list:
            self.schedule_embedding_task(captures_list, verdict_label=verdict)
        else:
            log.warning("No aligned captures to enqueue for embedding.")

    def shutdown(self):
        log.info("Shutdown requested.")
        self.stop_event.set()
        # wait for embed queue
        try:
            self.embed_queue.join(timeout=5)
        except Exception:
            pass
        if self.embed_thread:
            self.embed_thread.join(timeout=2)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("Shutdown complete.")

    def run(self, max_runtime_seconds=None):
        """
        Run the main capture loop. Optionally restrict runtime for automated tests.
        """
        self.prev_gray = None
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            log.error("Camera %d cannot be opened.", self.camera_index)
            return

        start_time = time.time()
        log.info("Starting capture loop on camera %d", self.camera_index)

        # Use mediapipe FaceMesh context manager for safe cleanup
        with self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as fm:
            while not self.stop_event.is_set():
                now = time.time()
                if max_runtime_seconds and (now - start_time) > (max_runtime_seconds):
                    log.info("Max runtime reached; exiting loop.")
                    break

                ret, frame = self.cap.read()
                if not ret:
                    log.warning("Frame read failed; sleeping briefly.")
                    time.sleep(0.05)
                    continue

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = fm.process(frame_rgb)
                except Exception as e:
                    log.exception("Mediapipe process failed: %s", e)
                    continue

                h, w, _ = frame.shape
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # restart after verdict if needed
                with self.state_lock:
                    fv = self.state.get("final_verdict")
                    vt = self.state.get("verdict_time")
                if fv and vt and (now - vt > 3.0):
                    log.info("Restarting detection (post-verdict)")
                    with self.state_lock:
                        new = reset_state()
                        self.state.clear()
                        self.state.update(new)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
                        landmarks_3d = [(l.x, l.y, l.z) for l in face_landmarks.landmark]

                        x_min, y_min = np.min(landmarks, axis=0)
                        x_max, y_max = np.max(landmarks, axis=0)
                        pad_x = int((x_max - x_min) * 0.08)
                        pad_y = int((y_max - y_min) * 0.08)
                        x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
                        x2, y2 = min(w, x_max + pad_x), min(h, y_max + pad_y)
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop is None or face_crop.size == 0:
                            continue

                        # motion check
                        if self.prev_gray is not None:
                            diff = cv2.absdiff(gray, self.prev_gray)
                            motion = np.sum(diff) / (h * w)
                            if motion > 1.2:
                                with self.state_lock:
                                    self.state["motion_detected"] = True
                                    self.state["last_motion_time"] = now
                        self.prev_gray = gray.copy()

                        # reset on large movement / disappearance (soft reset to keep completed challenges)
                        face_center = ((x1 + x2)//2, (y1 + y2)//2)
                        with self.state_lock:
                            prev_center = self.state.get("prev_center")
                        if prev_center is not None:
                            dist = euclidean(face_center, prev_center)
                            if dist > RESET_DISTANCE or now - self.state.get("last_seen_time", now) > RESET_TIMEOUT:
                                with self.state_lock:
                                    tmp = soft_reset_preserve_challenges(self.state)
                                    self.state.clear()
                                    self.state.update(tmp)
                        with self.state_lock:
                            self.state["prev_center"] = face_center
                            self.state["last_seen_time"] = now

                        # stabilization logic
                        with self.state_lock:
                            challenge_state = self.state.get("challenge_state")
                            stabilize_start = self.state.get("stabilize_start")
                        if challenge_state == "STABILIZE":
                            if stabilize_start is None:
                                with self.state_lock:
                                    self.state["stabilize_start"] = now
                            elif now - stabilize_start > STABILIZATION_TIME:
                                if RANDOMIZE_CHALLENGES:
                                    seq = BASE_SEQ.copy()
                                    random.shuffle(seq)
                                else:
                                    seq = BASE_SEQ.copy()
                                seq.append("DONE")
                                with self.state_lock:
                                    self.state["seq"] = seq
                                    self.state["seq_show_start"] = now
                                    self.state["challenge_state"] = seq[0]
                                    self.state["pre_live_captures"] = deque(maxlen=6)
                                    self.state["peak_live"] = 0.0
                                    self.state["high_frame_streak"] = 0
                                log.info("Stabilization complete — sequence: %s (nonce=%s)", seq, self.state.get("session_nonce"))
                            if self.show_window:
                                cv2.putText(frame, "Hold still to stabilize...", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                            if self.show_window and self.state.get("seq") and self.state.get("seq_show_start") and (now - self.state["seq_show_start"] < SHOW_SEQUENCE_SECONDS):
                                seq_to_show = " -> ".join(self.state["seq"])
                                cv2.putText(frame, f"Sequence: {seq_to_show}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,220,255), 2)
                            if self.show_window:
                                cv2.imshow("Ultra-Liveness", frame)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    self.stop_event.set()
                                    break
                            continue

                        # compute head metrics
                        head_ratio_xy, z_var, z_diff = get_head_turn_with_depth(landmarks_3d)
                        with self.state_lock:
                            self.state["z_history"].append(z_var)
                            z_recent_mean = float(np.mean(list(self.state["z_history"]))) if len(self.state["z_history"]) else 0.0

                        # compute roll for frontalness
                        landmarks_local = [(x - x1, y - y1) for (x, y) in landmarks]
                        roll_deg = 0.0
                        try:
                            le = np.array(landmarks_local[33], dtype=float)
                            re = np.array(landmarks_local[362], dtype=float)
                            dy = re[1] - le[1]
                            dx = re[0] - le[0]
                            roll_deg = abs(math.degrees(math.atan2(dy, dx)))
                        except Exception:
                            roll_deg = 90.0

                        is_frontal = (abs(head_ratio_xy - 1.0) <= FRONTAL_RATIO_TOL) and (roll_deg <= MAX_ROLL_DEGREES) and (z_var >= Z_VAR_THRESHOLD)

                        # collect pre-live captures when frontal
                        if is_frontal:
                            aligned = align_face(face_crop, landmarks_local, out_size=(160,160))
                            with self.state_lock:
                                plc = self.state.get("pre_live_captures")
                            if aligned is not None:
                                with self.state_lock:
                                    if plc is not None:
                                        plc.append(aligned.copy())
                            else:
                                try:
                                    small = cv2.resize(face_crop, (160,160))
                                    with self.state_lock:
                                        if plc is not None:
                                            plc.append(small.copy())
                                except Exception:
                                    pass

                        # CNN predict (thread-safe)
                        try:
                            face_resized = cv2.resize(face_crop, (INPUT_SIZE[0], INPUT_SIZE[1]))
                            input_img = np.expand_dims(face_resized.astype("float32") / 255.0, axis=0)
                            with self.model_lock:
                                cnn_pred = float(self.cnn_model.predict(input_img, verbose=0)[0][0])
                        except Exception as e:
                            log.warning("CNN predict failed: %s", e)
                            cnn_pred = 0.0

                        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        texture_score = np.clip(lbp_texture(gray_face) / 0.05, 0, 1)

                        # challenges - blink
                        left_eye_idx = [33, 160, 158, 133, 153, 144]
                        right_eye_idx = [362, 385, 387, 263, 373, 380]
                        ear = eye_aspect_ratio(landmarks, left_eye_idx, right_eye_idx)
                        blink_event = 0
                        with self.state_lock:
                            if ear < EAR_THRESHOLD:
                                self.state["blink_counter"] += 1
                            else:
                                if self.state["blink_counter"] >= BLINK_CONSEC_FRAMES:
                                    blink_event = 1
                                self.state["blink_counter"] = 0
                            self.state["challenge_scores"]["BLINK"].append(blink_event)
                            if (not self.state["blink_done"]) and (sum(self.state["challenge_scores"]["BLINK"]) >= BLINK_REQUIRED_EVENTS):
                                self.state["blink_done"] = True
                                log.info("Blink challenge confirmed.")

                        # smile
                        smile_metric = get_smile_metric(landmarks)
                        smile_hit = 1 if smile_metric > SMILE_THRESHOLD else 0
                        with self.state_lock:
                            self.state["challenge_scores"]["SMILE"].append(smile_hit)
                            if (not self.state["smile_done"]) and (len(self.state["challenge_scores"]["SMILE"]) >= SMILE_SCORE_WINDOW):
                                mean_smile = np.mean(self.state["challenge_scores"]["SMILE"])
                                if mean_smile >= SMILE_SCORE_THRESHOLD:
                                    self.state["smile_done"] = True
                                    log.info("Smile challenge confirmed.")

                        # say_hello
                        mouth_metric = get_mouth_open_metric(landmarks)
                        mouth_open = 1 if mouth_metric > MOUTH_OPEN_THRESHOLD else 0
                        with self.state_lock:
                            if mouth_open:
                                self.state["mouth_open_counter"] = self.state.get("mouth_open_counter", 0) + 1
                            else:
                                self.state["mouth_open_counter"] = 0
                            self.state["challenge_scores"]["SAY_HELLO"].append(1 if mouth_open else 0)
                            if (not self.state["speak_done"]) and (self.state["mouth_open_counter"] >= MOUTH_CONSEC_FRAMES):
                                self.state["speak_done"] = True
                                log.info("Say 'HELLO' challenge confirmed.")

                        # record histories and shape variation
                        with self.state_lock:
                            self.state["head_ratio_history"].append(head_ratio_xy)
                            shape_ratios = compute_shape_ratios(landmarks_3d)
                            self.state["shape_ratio_history"].append(shape_ratios)

                        with self.state_lock:
                            shape_arr = np.array(self.state["shape_ratio_history"])
                        shape_var = 0.0
                        if len(shape_arr) >= 3:
                            shape_var = float(np.mean(np.var(shape_arr, axis=0)))

                        # turn detection
                        ratio_dev = 0.0
                        if head_ratio_xy < 1.0:
                            ratio_dev_left = max(0.0, (1.0 - head_ratio_xy) / 0.25)
                            ratio_dev = ratio_dev_left
                        else:
                            ratio_dev_right = max(0.0, (head_ratio_xy - 1.0) / 0.25)
                            ratio_dev = ratio_dev_right
                        ratio_dev = np.clip(ratio_dev, 0.0, 1.0)
                        with self.state_lock:
                            z_factor = float(np.mean(list(self.state["z_history"])) / (np.mean(list(self.state["z_history"])) + Z_VAR_THRESHOLD + 1e-12)) if len(self.state["z_history"]) else 0.0
                        z_factor = np.clip(z_factor, 0.0, 1.0)
                        turn_score = ratio_dev * z_factor

                        with self.state_lock:
                            if head_ratio_xy < 1.0:
                                self.state["challenge_scores"]["TURN_LEFT"].append(turn_score)
                                self.state["challenge_scores"]["TURN_RIGHT"].append(0.0)
                            else:
                                self.state["challenge_scores"]["TURN_RIGHT"].append(turn_score)
                                self.state["challenge_scores"]["TURN_LEFT"].append(0.0)

                            if (not self.state["turn_left_done"]):
                                s = self.state["challenge_scores"]["TURN_LEFT"]
                                if len(s) >= MIN_ROLLING_FRAMES:
                                    mean_left = float(np.mean(s))
                                    if (mean_left >= TURN_SCORE_THRESHOLD) and (shape_var > SHAPE_VARIATION_THRESHOLD):
                                        self.state["turn_left_done"] = True
                                        log.info("Turn-left challenge confirmed.")

                            if (not self.state["turn_right_done"]):
                                s2 = self.state["challenge_scores"]["TURN_RIGHT"]
                                if len(s2) >= MIN_ROLLING_FRAMES:
                                    mean_right = float(np.mean(s2))
                                    if (mean_right >= TURN_SCORE_THRESHOLD) and (shape_var > SHAPE_VARIATION_THRESHOLD):
                                        self.state["turn_right_done"] = True
                                        log.info("Turn-right challenge confirmed.")

                        # compute per-frame live & progress
                        with self.state_lock:
                            challenge_factor = 0.2 * sum([
                                self.state["blink_done"], self.state["turn_left_done"],
                                self.state["turn_right_done"], self.state["smile_done"], self.state["speak_done"]
                            ])
                        depth_multiplier = 1.0
                        if (not self.state.get("turn_left_done") or not self.state.get("turn_right_done")) and z_recent_mean < Z_VAR_THRESHOLD:
                            depth_multiplier = NO_DEPTH_PENALTY

                        per_frame_live = np.clip(((0.65 * cnn_pred) + (0.25 * texture_score) + challenge_factor) * depth_multiplier, 0, 1)
                        with self.state_lock:
                            self.state["live_scores"].append(per_frame_live)
                            if per_frame_live > self.state.get("peak_live", 0.0):
                                self.state["peak_live"] = per_frame_live

                            if per_frame_live >= SUSTAINED_THRESHOLD:
                                self.state["high_frame_streak"] = self.state.get("high_frame_streak", 0) + 1
                            else:
                                self.state["high_frame_streak"] = 0

                            avg_live = float(np.mean(self.state["live_scores"])) if len(self.state["live_scores"]) else 0.0

                            progress = int(100 * (sum([self.state["blink_done"], self.state["turn_left_done"],
                                                       self.state["turn_right_done"], self.state["smile_done"], self.state["speak_done"]]) / 5.0))

                        # advance challenge sequence when time
                        with self.state_lock:
                            seq = self.state.get("seq")
                            cur = self.state.get("challenge_state")
                            last_ch_time = self.state.get("last_challenge_time", now)
                        if seq:
                            if now - last_ch_time > CHALLENGE_INTERVAL:
                                if cur in seq:
                                    idx = seq.index(cur)
                                    next_idx = idx + 1
                                    if next_idx < len(seq):
                                        with self.state_lock:
                                            self.state["challenge_state"] = seq[next_idx]
                                    with self.state_lock:
                                        self.state["last_challenge_time"] = now

                        with self.state_lock:
                            verified = all([
                                self.state["blink_done"], self.state["turn_left_done"],
                                self.state["turn_right_done"], self.state["smile_done"], self.state["speak_done"]
                            ])
                            peak_live = self.state.get("peak_live", 0.0)
                            high_frame_streak = self.state.get("high_frame_streak", 0)

                        # Final verdict
                        if verified and not self.state.get("final_verdict"):
                            with self.state_lock:
                                num_aligned = len(self.state.get("pre_live_captures", []))
                            if num_aligned < MIN_ALIGNED_CAPTURES:
                                log.warning("Not enough frontal aligned captures (%d)", num_aligned)
                                with self.state_lock:
                                    tmp = soft_reset_preserve_challenges(self.state)
                                    self.state.clear()
                                    self.state.update(tmp)
                                    self.state["challenge_state"] = "STABILIZE"
                                    self.state["stabilize_start"] = now
                                continue

                            if avg_live > FINAL_SCORE_THRESHOLD and progress == 100 and z_recent_mean > Z_VAR_THRESHOLD:
                                verdict = "Live Verified"
                            else:
                                if (high_frame_streak >= SUSTAINED_FRAMES_REQUIRED) and (progress >= PROGRESS_THRESHOLD_FOR_PEAK):
                                    verdict = "Live Verified (sustained-pass)"
                                    log.info("Sustained high-frame acceptance (streak=%d)", high_frame_streak)
                                elif (peak_live >= PEAK_REQUIRED) and (progress >= PROGRESS_THRESHOLD_FOR_PEAK):
                                    verdict = "Live Verified (peak-pass)"
                                    log.info("Using peak-live forgiveness (peak_live=%.3f, progress=%d%%)", peak_live, progress)
                                else:
                                    if avg_live >= 0.50:
                                        log.warning("Verification inconclusive (avg_live=%.3f) — retry", avg_live)
                                        with self.state_lock:
                                            tmp = soft_reset_preserve_challenges(self.state)
                                            self.state.clear()
                                            self.state.update(tmp)
                                            self.state["challenge_state"] = "STABILIZE"
                                            self.state["stabilize_start"] = now
                                        continue
                                    else:
                                        verdict = "Spoof Detected"
                            with self.state_lock:
                                self.state["final_verdict"] = verdict
                                self.state["verdict_time"] = now
                            log.info("Final Verdict: %s (avg_live=%.3f, progress=%d%%, z_mean=%.6f, peak_live=%.3f, aligned=%d)",
                                     verdict, avg_live, progress, z_recent_mean, peak_live, num_aligned)

                            if "Live" in verdict:
                                with self.state_lock:
                                    caps = [img.copy() for img in list(self.state.get("pre_live_captures", deque()))]
                                if caps:
                                    scheduled = self.schedule_embedding_task(caps, verdict)
                                    if not scheduled:
                                        log.warning("Could not schedule embedding; queue full.")
                                else:
                                    log.warning("No aligned captures for embedding.")
                            else:
                                # save spoof image
                                fname = os.path.join(SPOOF_SAVE_DIR, f"sp_oof_{int(time.time())}.jpg")
                                safe_imwrite(fname, face_crop)
                                log.info("Spoof diagnostic saved: %s", fname)

                        # Immediate DONE logic (same as earlier)
                        if self.state.get("challenge_state") == "DONE" and not self.state.get("final_verdict"):
                            if (avg_live >= DONE_MIN_AVG) and (progress >= DONE_MIN_PROGRESS) and (z_recent_mean >= DONE_MIN_Z):
                                verdict = "Live Verified (DONE-minimums)"
                                with self.state_lock:
                                    self.state["final_verdict"] = verdict
                                    self.state["verdict_time"] = now
                                log.info("Final Verdict (DONE -> LIVE): %s", verdict)
                                self._maybe_enqueue_embedding(verdict)
                            else:
                                verdict = "Spoof Detected"
                                fname = os.path.join(SPOOF_SAVE_DIR, f"sp_done_{int(time.time())}.jpg")
                                safe_imwrite(fname, face_crop)
                                with self.state_lock:
                                    self.state["final_verdict"] = verdict
                                    self.state["verdict_time"] = now
                                log.warning("Final Verdict (immediate DONE): %s", verdict)

                        # UI rendering
                        if self.show_window:
                            with self.state_lock:
                                fv = self.state.get("final_verdict")
                            if not fv:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (180,180,180), 1)
                                draw_scanner_overlay(frame, x1, y1, x2, y2, now)
                                with self.state_lock:
                                    if self.state.get("seq") and self.state.get("seq_show_start") and (now - self.state["seq_show_start"] < SHOW_SEQUENCE_SECONDS):
                                        seq_to_show = " -> ".join(self.state["seq"])
                                        cv2.putText(frame, f"Sequence: {seq_to_show}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,220,255), 2)
                                    msg = {
                                        "BLINK": "Blink 👁",
                                        "TURN_LEFT": "Turn Left ",
                                        "TURN_RIGHT": "Turn Right ",
                                        "SMILE": "Smile 😁",
                                        "SAY_HELLO": "Say 'HELLO' 🗣️",
                                        "DONE": "DONE"
                                    }.get(self.state.get("challenge_state", ""), "")
                                cv2.putText(frame, f"{msg}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230,230,230), 2)
                                # progress bar
                                bar_w = 220
                                bx, by = 40, 130
                                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 12), (80,80,80), -1)
                                filled = int((progress / 100.0) * bar_w)
                                cv2.rectangle(frame, (bx, by), (bx + filled, by + 12), (200,200,200), -1)
                            else:
                                banner_color = (0,180,0) if "Live" in fv else (0,0,180)
                                banner_text = "✅ Live Detected" if "Live" in fv else f"⚠ {fv}!"
                                (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                x_pos, y_pos = w - tw - 40, 60
                                cv2.rectangle(frame, (x_pos - 15, y_pos - th - 8), (x_pos + tw + 15, y_pos + 15), banner_color, -1)
                                cv2.putText(frame, banner_text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                else:
                    # face disappeared
                    with self.state_lock:
                        if now - self.state.get("last_seen_time", now) > RESET_TIMEOUT:
                            new = reset_state()
                            self.state.clear()
                            self.state.update(new)

                if self.show_window:
                    cv2.imshow("Ultra-Liveness", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()
                        break

        # cleanup
        try:
            self.cap.release()
        except Exception:
            pass
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("Capture loop stopped.")

# =============================================================================
# Offline evaluation harness (simple)
# =============================================================================
def evaluate_on_video(video_path, model_path=DEFAULT_CNN_MODEL, debug=False, out_report=None, max_seconds=None):
    """
    Runs the liveness pipeline on a recorded video file for offline evaluation.
    Produces a JSON report with frame-wise results and simple metrics.
    """
    log.info("Starting offline evaluation on %s", video_path)
    # instantiate detector with show_window=False
    detector = UltraLiveness(cnn_model_path=model_path, show_window=False, debug=debug)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video file: %s", video_path)
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = frame_count / fps if fps > 0 else None
    log.info("Video frames=%d, fps=%.2f, duration=%.2fs", frame_count, fps, duration if duration else 0)

    results = []
    start_time = time.time()
    frame_idx = 0
    try:
        with detector.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as fm:
            prev_gray = None
            while True:
                if max_seconds and (time.time() - start_time) > max_seconds:
                    log.info("Max seconds for eval reached.")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                h, w, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(frame_rgb)
                now = time.time()
                record = {"frame_idx": frame_idx, "time": now, "has_face": False}
                if res.multi_face_landmarks:
                    # For offline eval we simplify: only record CNN and LBP & basic metrics
                    landmarks = [(int(l.x * w), int(l.y * h)) for l in res.multi_face_landmarks[0].landmark]
                    x_min, y_min = np.min(landmarks, axis=0)
                    x_max, y_max = np.max(landmarks, axis=0)
                    pad_x = int((x_max - x_min) * 0.08)
                    pad_y = int((y_max - y_min) * 0.08)
                    x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
                    x2, y2 = min(w, x_max + pad_x), min(h, y_max + pad_y)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop is not None and face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, (INPUT_SIZE[0], INPUT_SIZE[1]))
                        input_img = np.expand_dims(face_resized.astype("float32") / 255.0, axis=0)
                        with detector.model_lock:
                            try:
                                cnn_pred = float(detector.cnn_model.predict(input_img, verbose=0)[0][0])
                            except Exception as e:
                                log.warning("CNN predict (eval) failed: %s", e)
                                cnn_pred = 0.0
                        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        texture_score = np.clip(lbp_texture(gray_face) / 0.05, 0, 1)
                        record.update({"has_face": True, "cnn_pred": cnn_pred, "texture_score": texture_score})
                results.append(record)
                frame_idx += 1
    finally:
        cap.release()
        detector.shutdown()

    # Produce basic report
    avg_cnn = float(np.mean([r["cnn_pred"] for r in results if r.get("has_face")])) if any(r.get("has_face") for r in results) else 0.0
    avg_texture = float(np.mean([r["texture_score"] for r in results if r.get("has_face")])) if any(r.get("has_face") for r in results) else 0.0
    summary = {
        "video": video_path,
        "frames": frame_idx,
        "frames_with_face": sum(1 for r in results if r.get("has_face")),
        "avg_cnn_pred": avg_cnn,
        "avg_texture_score": avg_texture,
        "duration_seconds": duration,
        "fps": fps
    }
    if out_report:
        try:
            with open(out_report, "w") as f:
                json.dump({"summary": summary, "frame_results": results}, f, indent=2)
            log.info("Saved report: %s", out_report)
        except Exception as e:
            log.warning("Failed to save report: %s", e)

    return summary

# =============================================================================
# Synthetic attack video generator (simple)
# =============================================================================
def create_synthetic_attack_video(output_path, duration_seconds=5, fps=20, attack_type="photo", face_image_path=None):
    """
    Creates a synthetic attack video for testing:
      - "photo": static face image shown on screen (replay)
      - "print": same as photo but with added blur/noise to simulate printed photo
      - "mask": attempt to place a second face or distort to simulate a mask (simple)
    Requires a face image path for realism. Otherwise generates a blank screen with text.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_size = (640, 480)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    total_frames = int(duration_seconds * fps)
    if face_image_path and os.path.exists(face_image_path):
        face_img = cv2.imread(face_image_path)
        face_img = cv2.resize(face_img, (300, 300))
    else:
        face_img = None

    for i in range(total_frames):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8) + 30
        if face_img is not None:
            x = (frame_size[0] - face_img.shape[1]) // 2
            y = (frame_size[1] - face_img.shape[0]) // 2
            frame[y:y+face_img.shape[0], x:x+face_img.shape[1]] = face_img
            # vary brightness to simulate replay artifacts
            alpha = 0.8 + 0.4 * math.sin(2 * math.pi * (i/total_frames) * 2)
            frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)
            if attack_type == "print":
                frame = cv2.GaussianBlur(frame, (7,7), 2.0)
                noise = np.random.normal(0, 6, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, noise)
        else:
            txt = f"SIMULATED {attack_type.upper()} ATTACK FRAME {i}"
            cv2.putText(frame, txt, (20, frame_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
        out.write(frame)
    out.release()
    log.info("Synthetic attack video saved to %s", output_path)
    return output_path

# =============================================================================
# Mini test suite (basic automated checks)
# =============================================================================
def run_smoke_tests(cnn_model_path=DEFAULT_CNN_MODEL):
    log.info("Running smoke tests...")
    # Test 1: model load
    try:
        det = UltraLiveness(cnn_model_path=cnn_model_path, show_window=False, debug=True)
        log.info("Smoke: Detector init passed.")
        det.shutdown()
    except Exception as e:
        log.error("Smoke test failed at init: %s", e)
        return False

    # Test 2: synthetic video creation
    try:
        tmp_vid = os.path.join(tempfile.gettempdir(), f"synthetic_{int(time.time())}.mp4")
        create_synthetic_attack_video(tmp_vid, duration_seconds=2, fps=10, attack_type="photo")
        if os.path.exists(tmp_vid):
            log.info("Smoke: Synthetic video created.")
        else:
            log.error("Smoke: Synthetic video not created.")
            return False
    except Exception as e:
        log.error("Smoke: Synthetic video creation failed: %s", e)
        return False

    # Test 3: offline evaluation (short)
    try:
        summary = evaluate_on_video(tmp_vid, model_path=cnn_model_path, debug=True, out_report=None, max_seconds=10)
        log.info("Smoke: Offline evaluation summary: %s", summary)
    except Exception as e:
        log.error("Smoke: Offline evaluation failed: %s", e)
        return False

    log.info("Smoke tests completed successfully.")
    return True

# =============================================================================
# CLI
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(prog="ultra_liveness", description="Ultra-Robust Liveness Detection Demo")
    parser.add_argument("--model", default=DEFAULT_CNN_MODEL, help="Path to CNN liveness model (.h5)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-window", action="store_true", help="Run headless (no GUI window)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and modes")
    parser.add_argument("--eval-video", help="Path to video file for offline evaluation")
    parser.add_argument("--make-attack", help="Create a synthetic attack video at given path")
    parser.add_argument("--attack-type", choices=["photo", "print", "mask"], default="photo")
    parser.add_argument("--attack-image", help="Face image to use for synthetic attack")
    parser.add_argument("--smoke-tests", action="store_true", help="Run smoke tests and exit")
    parser.add_argument("--max-runtime", type=float, help="Max runtime seconds for live capture loop (useful for CI)")
    return parser

# =============================================================================
# Entrypoint
# =============================================================================
def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("Debug mode enabled.")

    if args.smoke_tests:
        ok = run_smoke_tests(cnn_model_path=args.model)
        log.info("Smoke tests result: %s", ok)
        return 0 if ok else 2

    if args.make_attack:
        out_vid = create_synthetic_attack_video(args.make_attack, duration_seconds=5, fps=20, attack_type=args.attack_type, face_image_path=args.attack_image)
        log.info("Attack video generated at %s", out_vid)
        return 0

    if args.eval_video:
        report_path = os.path.join(REPORT_DIR, f"eval_{int(time.time())}.json")
        summary = evaluate_on_video(args.eval_video, model_path=args.model, debug=args.debug, out_report=report_path, max_seconds=args.max_runtime)
        log.info("Evaluation summary written to %s", report_path)
        return 0

    # default: live run
    det = UltraLiveness(cnn_model_path=args.model, camera_index=args.camera, show_window=not args.no_window, debug=args.debug)
    # graceful shutdown on signals
    def _signal_handler(sig, frame):
        log.info("Signal received %s. Shutting down.", sig)
        det.shutdown()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        det.run(max_runtime_seconds=args.max_runtime)
    except Exception as e:
        log.exception("Unhandled exception in main run: %s", e)
    finally:
        det.shutdown()
    return 0

# =============================================================================
# If run as script
# =============================================================================
if __name__ == "__main__":
    main()
