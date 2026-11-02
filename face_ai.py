"""Deep-learning face recognition (OpenCV Zoo models, run with OpenCV's built-in DNN).

* YuNet  (models/face_detection_yunet_2023mar.onnx)  finds faces + 5 landmarks (eyes, nose, mouth).
* SFace  (models/face_recognition_sface_2021dec.onnx) turns an aligned face into a 128-number
  "face fingerprint" (embedding). Two fingerprints of the same person have a high cosine
  similarity; OpenCV's recommended "same person" threshold is 0.363.

Used for:
  * ID card photo vs live face          (is this really your card?)
  * duplicate detection at registration (is this person already a voter?)
  * a second, stronger check at the voting booth

Fingerprints are stored in data/embeddings.pkl ({voter_id: vector}). Voters registered before
this existed get theirs computed from their stored face samples the first time they're needed.
"""

import os
import pickle
import threading

import cv2
import numpy as np

MODELS_DIR = "models"
DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
EMBEDDINGS_PATH = os.path.join("data", "embeddings.pkl")
PENDING_DIR = os.path.join("data", "pending")

# Cosine similarity thresholds (SFace).
SAME_PERSON = float(os.environ.get("AI_SAME_PERSON", "0.363"))     # OpenCV's recommended value
CLEARLY_SAME = float(os.environ.get("AI_CLEARLY_SAME", "0.55"))     # duplicate: refuse automatically
CLEARLY_DIFFERENT = float(os.environ.get("AI_CLEARLY_DIFFERENT", "0.20"))  # ID card: refuse automatically

_lock = threading.Lock()          # OpenCV DNN models are not thread-safe
_models = {}
_store = {"loaded": False, "vectors": {}}


def available():
    return os.path.exists(DETECTOR_PATH) and os.path.exists(RECOGNIZER_PATH)


def _get(name):
    if name not in _models:
        if name == "detector":
            _models[name] = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320), 0.6, 0.3, 5000)
        else:
            _models[name] = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
    return _models[name]


def detect(image, min_score=0.6):
    """All faces in the image as YuNet rows: x, y, w, h, 5 landmark (x, y) pairs, score."""
    with _lock:
        detector = _get("detector")
        detector.setScoreThreshold(min_score)
        detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = detector.detect(image)
    return [] if faces is None else list(faces)


def _embed_face(image, face):
    with _lock:
        recognizer = _get("recognizer")
        vector = recognizer.feature(recognizer.alignCrop(image, face)).flatten().astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-9)


def embed_largest(image, min_score=0.6):
    """Fingerprint of the largest face in the image, or None if no face is found."""
    faces = detect(image, min_score)
    if not faces:
        return None
    return _embed_face(image, max(faces, key=lambda f: f[2] * f[3]))


def embed_many(images, limit=10):
    """Average fingerprint over several frames of the same person (more stable than one frame)."""
    step = max(1, len(images) // limit)
    vectors = [v for v in (embed_largest(image) for image in images[::step][:limit]) if v is not None]
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


def similarity(a, b):
    return float(np.dot(a, b))


def embed_from_samples(samples, limit=10):
    """Fingerprint from stored 50x50 face crops (for voters registered before the AI model)."""
    images = []
    for sample in samples[:: max(1, len(samples) // limit)][:limit]:
        face = cv2.resize(np.asarray(sample, np.uint8).reshape(50, 50, 3), (160, 160), interpolation=cv2.INTER_CUBIC)
        images.append(cv2.copyMakeBorder(face, 60, 60, 60, 60, cv2.BORDER_REPLICATE))
    return embed_many(images, limit)


# --------------------------------------------------------------------------
# Fingerprint store
# --------------------------------------------------------------------------

def _load():
    if not _store["loaded"]:
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, "rb") as file:
                _store["vectors"] = pickle.load(file)
        _store["loaded"] = True
    return _store["vectors"]


def _save():
    temp = EMBEDDINGS_PATH + ".tmp"
    with open(temp, "wb") as file:
        pickle.dump(_store["vectors"], file)
    os.replace(temp, EMBEDDINGS_PATH)


def set_embedding(voter_id, vector):
    _load()[voter_id] = vector
    _save()


def remove_embedding(voter_id):
    if _load().pop(voter_id, None) is not None:
        _save()


def get_embedding(voter_id):
    return _load().get(voter_id)


def backfill(face_dataset_reader):
    """Compute fingerprints for voters who only have old-style face samples."""
    vectors = _load()
    faces, labels = face_dataset_reader()
    labels = list(labels)
    missing = [v for v in dict.fromkeys(labels) if v not in vectors]
    for voter_id in missing:
        samples = [faces[i] for i, label in enumerate(labels) if label == voter_id]
        vector = embed_from_samples(samples)
        if vector is not None:
            vectors[voter_id] = vector
    if missing:
        _save()


def save_pending(ref, vector):
    os.makedirs(PENDING_DIR, exist_ok=True)
    np.save(os.path.join(PENDING_DIR, f"{ref}_emb.npy"), vector)


def discard_pending(ref):
    path = os.path.join(PENDING_DIR, f"{ref}_emb.npy")
    if os.path.exists(path):
        os.remove(path)


def approve_pending(ref):
    path = os.path.join(PENDING_DIR, f"{ref}_emb.npy")
    if os.path.exists(path):
        set_embedding(ref, np.load(path))
        os.remove(path)


def _pending_vectors(exclude=None):
    vectors = {}
    if os.path.isdir(PENDING_DIR):
        for name in os.listdir(PENDING_DIR):
            if name.endswith("_emb.npy") and name[:-8] != exclude:
                vectors[name[:-8]] = np.load(os.path.join(PENDING_DIR, name))
    return vectors


def best_match(vector, exclude=None, include_pending=True):
    """(voter_id, similarity) of the most similar registered voter / applicant, or (None, -1)."""
    candidates = dict(_load())
    if include_pending:
        candidates.update(_pending_vectors(exclude))
    candidates.pop(exclude, None)
    if not candidates:
        return None, -1.0
    ids = list(candidates)
    scores = np.stack([candidates[i] for i in ids]) @ vector
    best = int(np.argmax(scores))
    return ids[best], float(scores[best])


def top_two(vector):
    """Best and second-best registered voter (for confident identification at the booth)."""
    vectors = _load()
    if not vectors:
        return []
    ids = list(vectors)
    scores = np.stack([vectors[i] for i in ids]) @ vector
    order = np.argsort(scores)[::-1][:2]
    return [(ids[i], float(scores[i])) for i in order]
