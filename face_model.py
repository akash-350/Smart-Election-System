"""Shared face detection, recognition, liveness and face-dataset storage.

Face samples are stored as flattened 50x50 BGR crops in data/faces_data.pkl,
with one label per sample in data/names.pkl. A larger photo of each voter is
kept in data/photos/<voter_id>.jpg for the admin voter list.

Recognition uses k-nearest-neighbours on lighting-normalised grayscale
features. A face is only accepted when most of its k nearest samples belong to
the same voter AND they are close enough (FACE_MATCH_THRESHOLD), so an
unregistered person is rejected instead of being mapped to the nearest voter.

Liveness: the voter is asked to close and then open their eyes. A printed
photo can't do that. (A video replay still could — this is a classroom demo.)
"""

import base64
import binascii
import os
import pickle
import threading
from collections import Counter

import cv2
import numpy as np

DATA_DIR = "data"
FACES_PATH = os.path.join(DATA_DIR, "faces_data.pkl")
NAMES_PATH = os.path.join(DATA_DIR, "names.pkl")
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")

FACE_SIZE = (50, 50)
PHOTO_SIZE = 160
MIN_SAMPLES_PER_VOTER = 30
K_NEIGHBOURS = 5
MIN_NEIGHBOUR_AGREEMENT = 4

# Distance between unit-normalised feature vectors: 0 = identical,
# ~1.41 = unrelated. 0.85 corresponds to a pixel correlation of about 0.64.
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.85"))
LIVENESS_REQUIRED = os.environ.get("LIVENESS_CHECK", "on").lower() not in ("0", "off", "false", "no")

_lock = threading.Lock()
_cache = {"key": None, "labels": None, "features": None}
_detectors = {}


class FaceDatasetError(Exception):
    """The stored face dataset is missing, invalid or inconsistent."""


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def _cascade(name):
    if name not in _detectors:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if detector.empty():
            raise RuntimeError(f"OpenCV detector {name} could not be loaded.")
        _detectors[name] = detector
    return _detectors[name]


def get_detector():
    return _cascade("haarcascade_frontalface_default.xml")


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return get_detector().detectMultiScale(gray, 1.3, 5)


def crop_face(image, box, size=FACE_SIZE):
    x, y, w, h = box
    return cv2.resize(image[y:y + h, x:x + w], size)


def crop_photo(image, box):
    """A slightly wider square crop for the voter's profile photo."""
    x, y, w, h = box
    margin = int(w * 0.25)
    height, width = image.shape[:2]
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, width), min(y + h + margin, height)
    return cv2.resize(image[y0:y1, x0:x1], (PHOTO_SIZE, PHOTO_SIZE))


def detect_single_face(image):
    """Return (face_crop, number_of_faces). face_crop is None unless exactly one face."""
    boxes = detect_faces(image)
    if len(boxes) != 1:
        return None, len(boxes)
    return crop_face(image, boxes[0]), 1


def detect_eyes(image, box):
    """Eyes found in the upper part of a face, as boxes relative to the face (0-1)."""
    x, y, w, h = box
    gray = cv2.cvtColor(image[y:y + int(h * 0.6), x:x + w], cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_size = max(int(w / 8), 12)
    eyes = _cascade("haarcascade_eye.xml").detectMultiScale(gray, 1.1, 6, minSize=(min_size, min_size))
    return [(ex / w, ey / h, ew / w, eh / h) for ex, ey, ew, eh in eyes]


def count_eyes(image, box):
    """Number of eyes the Haar detector finds (it often also "finds" closed eyes)."""
    return len(detect_eyes(image, box))


def decode_data_url(data_url):
    """Decode a base64 'data:image/...' URL into a BGR image, or return None."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except (IndexError, ValueError, binascii.Error):
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------
# Features and matching
# --------------------------------------------------------------------------

def to_features(samples):
    """Convert flattened 50x50 BGR samples into unit-length normalised vectors."""
    samples = np.asarray(samples, dtype=np.uint8).reshape(-1, FACE_SIZE[1], FACE_SIZE[0], 3)
    features = np.empty((len(samples), FACE_SIZE[0] * FACE_SIZE[1]), dtype=np.float32)
    for i, sample in enumerate(samples):
        gray = cv2.equalizeHist(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY))
        vector = gray.astype(np.float32).ravel()
        vector -= vector.mean()
        norm = np.linalg.norm(vector)
        features[i] = vector / norm if norm > 0 else vector
    return features


def _match(features, labels, query):
    """Core KNN decision on prepared features. Returns (voter_id or None, reason, distance)."""
    distances = np.linalg.norm(features - query, axis=1)
    nearest = np.argsort(distances)[:K_NEIGHBOURS]
    voter_id, votes = Counter(labels[nearest]).most_common(1)[0]
    voter_distance = float(distances[nearest][labels[nearest] == voter_id].mean())
    if votes < MIN_NEIGHBOUR_AGREEMENT:
        return None, "uncertain", voter_distance
    if voter_distance > MATCH_THRESHOLD:
        return None, "unknown", voter_distance
    return str(voter_id), None, voter_distance


REASONS = {
    "uncertain": "Face match is uncertain.",
    "unknown": "Face not recognised as a registered voter.",
}


# --------------------------------------------------------------------------
# Dataset storage
# --------------------------------------------------------------------------

def _read_dataset():
    if not os.path.exists(NAMES_PATH) and not os.path.exists(FACES_PATH):
        return np.empty((0, FACE_SIZE[0] * FACE_SIZE[1] * 3), dtype=np.uint8), []
    if not (os.path.exists(NAMES_PATH) and os.path.exists(FACES_PATH)):
        raise FaceDatasetError("Face dataset is incomplete (one of the .pkl files is missing).")

    with open(FACES_PATH, "rb") as file:
        faces = np.asarray(pickle.load(file))
    with open(NAMES_PATH, "rb") as file:
        labels = list(pickle.load(file))

    if len(faces) != len(labels):
        raise FaceDatasetError("Face dataset mismatch between faces and names.")
    return faces, labels


def _write_dataset(faces, labels):
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_faces, temp_names = FACES_PATH + ".tmp", NAMES_PATH + ".tmp"
    try:
        with open(temp_faces, "wb") as file:
            pickle.dump(np.asarray(faces, dtype=np.uint8), file)
        with open(temp_names, "wb") as file:
            pickle.dump(list(labels), file)
        os.replace(temp_faces, FACES_PATH)
        os.replace(temp_names, NAMES_PATH)
    finally:
        for path in (temp_faces, temp_names):
            if os.path.exists(path):
                os.remove(path)
    _cache["key"] = None


def _dataset_key():
    try:
        return (os.stat(FACES_PATH).st_mtime_ns, os.stat(NAMES_PATH).st_mtime_ns)
    except FileNotFoundError:
        return None


def _get_model():
    """Return (labels array, features), reloading only when the files change."""
    with _lock:
        key = _dataset_key()
        if key is None:
            raise FaceDatasetError("No registered voters found.")
        if _cache["key"] != key:
            faces, labels = _read_dataset()
            if len(labels) < K_NEIGHBOURS:
                raise FaceDatasetError("Not enough face samples registered.")
            _cache.update(key=key, labels=np.asarray(labels), features=to_features(faces))
        return _cache["labels"], _cache["features"]


def load_model():
    """Load (or validate) the face dataset and detectors up front."""
    get_detector()
    _cascade("haarcascade_eye.xml")
    _get_model()


def registered_ids():
    _, labels = _read_dataset()
    return set(labels)


def save_voter_samples(voter_id, face_crops, replace=False):
    """Add a voter's face samples (or replace them when re-capturing)."""
    if len(face_crops) < MIN_SAMPLES_PER_VOTER:
        raise ValueError(
            f"Only {len(face_crops)} valid face samples. "
            f"At least {MIN_SAMPLES_PER_VOTER} are required."
        )
    new_faces = np.asarray(face_crops, dtype=np.uint8).reshape(len(face_crops), -1)

    with _lock:
        faces, labels = _read_dataset()
        if voter_id in labels:
            if not replace:
                raise ValueError("Voter ID already exists in face dataset.")
            keep = np.asarray([label != voter_id for label in labels], dtype=bool)
            faces = faces[keep]
            labels = [label for label in labels if label != voter_id]

        updated_faces = np.concatenate((faces, new_faces), axis=0) if len(faces) else new_faces
        _write_dataset(updated_faces, labels + [voter_id] * len(new_faces))


def remove_voter_samples(voter_id):
    with _lock:
        faces, labels = _read_dataset()
        keep = np.asarray([label != voter_id for label in labels], dtype=bool)
        if len(labels) and not keep.all():
            _write_dataset(faces[keep], [label for label in labels if label != voter_id])
    photo = photo_path(voter_id)
    if os.path.exists(photo):
        os.remove(photo)


# --------------------------------------------------------------------------
# Pending applicants (kept OUT of the voting dataset until approved)
# --------------------------------------------------------------------------

PENDING_DIR = os.path.join(DATA_DIR, "pending")


def _pending_path(ref):
    return os.path.join(PENDING_DIR, f"{ref}.npy")


def save_pending_samples(ref, face_crops):
    os.makedirs(PENDING_DIR, exist_ok=True)
    np.save(_pending_path(ref), np.asarray(face_crops, dtype=np.uint8).reshape(len(face_crops), -1))


def discard_pending_samples(ref):
    path = _pending_path(ref)
    if os.path.exists(path):
        os.remove(path)


def approve_pending_samples(ref):
    """Move an approved applicant's samples into the voting dataset."""
    path = _pending_path(ref)
    if not os.path.exists(path):
        raise FaceDatasetError("Face samples for this application are missing.")
    save_voter_samples(ref, list(np.load(path)))
    os.remove(path)


def _pending_sets(exclude=None):
    sets = []
    if os.path.isdir(PENDING_DIR):
        for name in os.listdir(PENDING_DIR):
            ref = name[:-4]
            if name.endswith(".npy") and ref != exclude:
                sets.append((ref, np.load(os.path.join(PENDING_DIR, name))))
    return sets


def find_duplicate(face_crops, exclude=None):
    """Does this face already belong to a registered voter or another applicant?

    Returns (label, percent_of_samples_matched) or (None, 0).
    """
    faces, labels = _read_dataset()
    feature_parts, label_parts = [], []
    if len(labels):
        feature_parts.append(to_features(faces))
        label_parts += [str(label) for label in labels]
    for ref, samples in _pending_sets(exclude):
        feature_parts.append(to_features(samples))
        label_parts += [ref] * len(samples)
    if len(label_parts) < K_NEIGHBOURS:
        return None, 0

    features = np.concatenate(feature_parts)
    labels_array = np.asarray(label_parts)
    hits = Counter()
    queries = to_features(face_crops)
    for query in queries:
        match, _, _ = _match(features, labels_array, query)
        if match:
            hits[match] += 1
    if not hits:
        return None, 0
    label, count = hits.most_common(1)[0]
    percent = round(count / len(queries) * 100)
    return (label, percent) if percent >= 50 else (None, percent)


def face_quality(images_with_boxes):
    """Average brightness (0-255) and sharpness (Laplacian variance) of the face area."""
    brightness, sharpness = [], []
    for image, box in images_with_boxes:
        gray = cv2.cvtColor(crop_photo(image, box), cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    if not brightness:
        return None, None
    return round(float(np.median(brightness)), 1), round(float(np.median(sharpness)), 1)


# --------------------------------------------------------------------------
# Voter photos
# --------------------------------------------------------------------------

def photo_path(voter_id):
    return os.path.join(PHOTOS_DIR, f"{voter_id}.jpg")


def save_photo(voter_id, photo):
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    cv2.imwrite(photo_path(voter_id), photo, [cv2.IMWRITE_JPEG_QUALITY, 85])


def photo_bytes(voter_id):
    """JPEG bytes for a voter: saved photo, else an upscaled stored face sample."""
    path = photo_path(voter_id)
    if os.path.exists(path):
        with open(path, "rb") as file:
            return file.read()
    faces, labels = _read_dataset()
    for face, label in zip(faces, labels):
        if label == voter_id:
            sample = np.asarray(face, dtype=np.uint8).reshape(FACE_SIZE[1], FACE_SIZE[0], 3)
            image = cv2.resize(sample, (PHOTO_SIZE, PHOTO_SIZE), interpolation=cv2.INTER_CUBIC)
            return cv2.imencode(".jpg", image)[1].tobytes()
    return None


# --------------------------------------------------------------------------
# Identification and liveness
# --------------------------------------------------------------------------

def identify(face_crop):
    """Return (voter_id, None) on a confident match, else (None, reason text)."""
    labels, features = _get_model()
    voter_id, reason, _ = _match(features, labels, to_features([face_crop])[0])
    return voter_id, REASONS.get(reason)


def _analyse(images):
    """For each frame: (face_box or None, face_count)."""
    results = []
    for image in images:
        boxes = detect_faces(image)
        results.append((boxes[0] if len(boxes) == 1 else None, len(boxes)))
    return results


def identify_frames(images, min_agreement=0.8):
    """Identify one voter from several frames; every recognised frame must agree.

    Returns (voter_id, None, None) or (None, message, code).
    """
    results = []
    for image, (box, count) in zip(images, _analyse(images)):
        if box is None:
            if count > 1:
                return None, "Only one person should be in front of the camera.", "multiple_faces"
            continue
        voter_id, _ = identify(crop_face(image, box))
        results.append(voter_id)

    needed = max(1, round(len(images) * min_agreement))
    if len(results) < needed:
        return None, "Show your face clearly to the camera.", "no_face"
    matched = [r for r in results if r is not None]
    if len(set(matched)) > 1:
        return None, "Face match was inconsistent. Please try again.", "inconsistent"
    if len(matched) < needed:
        return None, "Face not recognised as a registered voter.", "not_recognised"
    return matched[0], None, None


def eye_states(images):
    """Open-eye counts for frames where exactly one face is visible."""
    states = []
    for image, (box, _) in zip(images, _analyse(images)):
        if box is not None:
            states.append(count_eyes(image, box))
    return states


# How much the eye area must change (vs. normal frame-to-frame noise) to count as "eyes closed".
LIVENESS_MIN_CHANGE = float(os.environ.get("LIVENESS_MIN_CHANGE", "1.8"))
EYE_NOISE_FLOOR = 0.12
# Dark-pupil count: "very dark" = the darkest DARK_PERCENT % of eye pixels while eyes are open.
DARK_PERCENT = 12.0
DARK_CLOSED_MAX = float(os.environ.get("LIVENESS_DARK_CLOSED_MAX", "0.62"))  # closed: at most 62% as many
DARK_REOPEN_MIN = 0.65                                                        # reopened: at least 65% back


def eye_band(image, box, size=(96, 32)):
    """The strip of the face around both eyes, aligned to the face box and lighting-normalised."""
    x, y, w, h = box
    y0, y1 = y + int(h * 0.18), y + int(h * 0.52)
    x0, x1 = x + int(w * 0.10), x + int(w * 0.90)
    band = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    band = cv2.resize(band, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    band = cv2.GaussianBlur(band, (3, 3), 0)
    return (band - band.mean()) / (band.std() + 1e-6)


def locate_eyes(faces):
    """Where each eye sits inside the face box, learned from eyes-open frames.

    Returns up to two relative boxes (left eye, right eye), or [] if no eyes were found.
    Measuring only these small boxes leaves out eyebrows and hair, which stay dark when
    the eyes close and would hide the change.
    """
    left, right = [], []
    for image, box in faces:
        for rx, ry, rw, rh in detect_eyes(image, box):
            if ry + rh / 2 > 0.55 or rw > 0.45:      # ignore nostrils / mouth / huge boxes
                continue
            (left if rx + rw / 2 < 0.5 else right).append((rx, ry, rw, rh))
    return [tuple(np.median(side, axis=0)) for side in (left, right) if len(side) >= max(2, len(faces) // 3)]


def eye_pixels(image, box, eyes=None):
    """Eye pixels divided by the face's average brightness (so exposure changes don't matter).

    With `eyes` (from locate_eyes) only the eye boxes are used — skipping the eyebrow at the
    top of each box. Without it, a strip across both eyes is used.
    """
    x, y, w, h = box
    face = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if eyes:
        crops = []
        for rx, ry, rw, rh in eyes:
            y0, y1 = int((ry + rh * 0.30) * h), int((ry + rh * 0.85) * h)
            x0, x1 = int((rx + rw * 0.12) * w), int((rx + rw * 0.88) * w)
            crop = face[max(y0, 0):max(y1, y0 + 2), max(x0, 0):max(x1, x0 + 2)]
            crops.append(cv2.resize(crop, (32, 16), interpolation=cv2.INTER_AREA))
        region = np.hstack(crops)
    else:
        region = cv2.resize(face[int(h * 0.25):int(h * 0.50), int(w * 0.12):int(w * 0.88)], (96, 24),
                            interpolation=cv2.INTER_AREA)
    return region / (face.mean() + 1e-6)


def _faces(images):
    """(image, face box) for every frame with exactly one face."""
    return [(image, box) for image, (box, _) in zip(images, _analyse(images)) if box is not None]


def _eye_frames(faces, eyes):
    """(eye band, Haar eye count, eye pixels) per face frame."""
    return [(eye_band(image, box), count_eyes(image, box), eye_pixels(image, box, eyes)) for image, box in faces]


def check_liveness(closed_images, open_images, reopened_images):
    """A live person's eyes change when they close them and change back when they open
    them. A printed photo doesn't change, even if it's moved around.

    Three signals. Pass = the dark-pupil check passes, OR the other two both agree:
      1. Dark-pupil count: open eyes have very dark pupils/irises. When eyes close, the number
         of very dark pixels around the eyes drops sharply, then comes back. This ignores
         movement, so waving a photo doesn't fake it.
      2. Eye-area change: the eye strip in the closed frames differs from both eyes-open
         phases clearly more than normal noise, while the two open phases match.
      3. Haar eye detector: eyes found in the open phases but not in the closed phase.
    Signals 2 and 3 are weaker on their own (a moved photo can sometimes fool one of them).

    Returns (ok, message, code, metrics) — metrics are logged to help tuning.
    """
    open_faces = _faces(open_images)
    eyes = locate_eyes(open_faces)
    opened = _eye_frames(open_faces, eyes)
    closed = _eye_frames(_faces(closed_images), eyes)
    reopened = _eye_frames(_faces(reopened_images), eyes)
    metrics = {"open_frames": len(opened), "closed_frames": len(closed), "reopen_frames": len(reopened),
               "eyes_located": len(eyes)}
    if len(opened) < 2 or len(closed) < max(3, len(closed_images) // 2) or not reopened:
        return False, "Keep your face in view while closing your eyes.", "liveness_face", metrics

    # --- 1. dark-pupil count ---------------------------------------------------
    open_pixels = np.concatenate([pixels.ravel() for _, _, pixels in opened])
    dark_level = float(np.percentile(open_pixels, DARK_PERCENT))   # "very dark" = darkest few % when open
    dark = lambda pixels: float(np.mean(pixels < dark_level)) / (DARK_PERCENT / 100)   # 1.0 = like open eyes
    closed_dark = float(np.mean(sorted(dark(p) for _, _, p in closed)[:3]))    # the 3 most "closed" frames
    # People open their eyes a moment late, so judge "open again" from the last 3 frames.
    reopen_dark = float(np.median([dark(p) for _, _, p in reopened][-3:]))
    dark_ok = closed_dark <= DARK_CLOSED_MAX and reopen_dark >= DARK_REOPEN_MIN

    # --- 2. eye-area change ----------------------------------------------------
    def dist(a, b):
        return float(np.mean(np.abs(a - b)))

    open_bands = [band for band, _, _ in opened]
    before = np.median(open_bands, axis=0)
    after = np.median([band for band, _, _ in reopened], axis=0)
    leave_one_out = [dist(open_bands[i], np.median(open_bands[:i] + open_bands[i + 1:], axis=0))
                     for i in range(len(open_bands))]
    noise = max(float(np.percentile(leave_one_out, 80)), EYE_NOISE_FLOOR)
    top3 = lambda values: float(np.mean(sorted(values, reverse=True)[:3]))
    closed_change = min(top3([dist(band, before) for band, _, _ in closed]),
                        top3([dist(band, after) for band, _, _ in closed])) / noise
    reopen_change = dist(before, after) / noise
    band_ok = (closed_change >= LIVENESS_MIN_CHANGE
               and reopen_change <= max(LIVENESS_MIN_CHANGE * 0.8, closed_change * 0.55))

    # --- 3. Haar eye detector --------------------------------------------------
    steady_rate = sum(1 for _, eyes, _ in opened if eyes >= 1) / len(opened)
    reopen_rate = sum(1 for _, eyes, _ in reopened if eyes >= 1) / len(reopened)
    closed_rate = sum(1 for _, eyes, _ in closed if eyes >= 1) / len(closed)
    haar_ok = steady_rate >= 0.6 and reopen_rate >= 0.5 and closed_rate <= 0.2

    metrics["dark_per_frame"] = {
        "open": [round(dark(p), 2) for _, _, p in opened],
        "closed": [round(dark(p), 2) for _, _, p in closed],
        "reopened": [round(dark(p), 2) for _, _, p in reopened],
    }
    metrics.update(closed_dark=round(closed_dark, 2), reopen_dark=round(reopen_dark, 2),
                   closed_change=round(closed_change, 2), reopen_change=round(reopen_change, 2),
                   haar_rates=(round(steady_rate, 2), round(closed_rate, 2), round(reopen_rate, 2)),
                   dark_ok=dark_ok, band_ok=band_ok, haar_ok=haar_ok)
    if dark_ok or (band_ok and haar_ok):
        return True, None, None, metrics
    if closed_dark <= DARK_CLOSED_MAX:
        return False, "Please open your eyes again when you hear the second beep.", "liveness_open", metrics
    return False, ("We couldn't see your eyes close. Close them firmly at the first beep "
                   "and keep them closed until the second beep."), "liveness_closed", metrics


# --------------------------------------------------------------------------
# Accuracy evaluation (for the report)
# --------------------------------------------------------------------------

def evaluate(window=3):
    """Measure matching quality on the stored samples.

    * Genuine test: each sample is matched with itself and its `window`
      neighbouring samples on both sides removed (so near-identical frames
      don't make the score look better than it is).
    * Impostor test: each voter is removed entirely, and their samples are
      matched against everyone else. Any match is a false acceptance.
    """
    faces, labels = _read_dataset()
    if len(labels) < K_NEIGHBOURS + 2 * window + 1:
        raise FaceDatasetError("Not enough face samples to evaluate.")
    labels = np.asarray(labels)
    features = to_features(faces)
    voters = list(dict.fromkeys(labels.tolist()))

    per_voter = {v: {"voter_id": v, "samples": 0, "correct": 0, "rejected": 0, "wrong": 0,
                     "impostor_tests": 0, "false_accepts": 0, "false_accept_as": Counter()} for v in voters}
    genuine_distances, impostor_distances = [], []

    for i in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        same = np.where(labels == labels[i])[0]
        position = int(np.where(same == i)[0][0])
        mask[same[max(0, position - window): position + window + 1]] = False
        if mask.sum() < K_NEIGHBOURS:
            continue
        voter_id, _, distance = _match(features[mask], labels[mask], features[i])
        stats = per_voter[labels[i]]
        stats["samples"] += 1
        genuine_distances.append(distance)
        if voter_id == labels[i]:
            stats["correct"] += 1
        elif voter_id is None:
            stats["rejected"] += 1
        else:
            stats["wrong"] += 1

    for voter in voters:
        mask = labels != voter
        if mask.sum() < K_NEIGHBOURS:
            continue
        for i in np.where(~mask)[0]:
            matched, _, distance = _match(features[mask], labels[mask], features[i])
            stats = per_voter[voter]
            stats["impostor_tests"] += 1
            impostor_distances.append(distance)
            if matched is not None:
                stats["false_accepts"] += 1
                stats["false_accept_as"][matched] += 1

    def pct(part, whole):
        return round(part / whole * 100, 1) if whole else None

    rows = []
    for stats in per_voter.values():
        stats["accuracy"] = pct(stats["correct"], stats["samples"])
        stats["far"] = pct(stats["false_accepts"], stats["impostor_tests"])
        stats["false_accept_as"] = ", ".join(f"{k} ({v})" for k, v in stats["false_accept_as"].most_common())
        rows.append(stats)

    total = sum(r["samples"] for r in rows)
    impostor_total = sum(r["impostor_tests"] for r in rows)
    return {
        "voters": rows,
        "threshold": MATCH_THRESHOLD,
        "samples": int(len(labels)),
        "genuine_accept_rate": pct(sum(r["correct"] for r in rows), total),
        "false_reject_rate": pct(sum(r["rejected"] for r in rows), total),
        "misidentification_rate": pct(sum(r["wrong"] for r in rows), total),
        "false_accept_rate": pct(sum(r["false_accepts"] for r in rows), impostor_total),
        "impostor_tests": impostor_total,
        "genuine_distance": round(float(np.median(genuine_distances)), 3) if genuine_distances else None,
        "impostor_distance": round(float(np.median(impostor_distances)), 3) if impostor_distances else None,
        "window": window,
    }
