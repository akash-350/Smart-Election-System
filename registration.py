"""Voter self-registration: validation and automatic verification checks.

Privacy: Aadhaar / Voter ID numbers are never stored in full. We keep a masked
copy for display (XXXX XXXX 1234) and a keyed HMAC-SHA256 hash, which is enough
to spot the same number being used twice but can't be turned back into it
without the secret key in data/.id_pepper.

What the checks can NOT do: confirm with UIDAI or the Election Commission that
an ID really exists (that needs official API access). They check the format,
the checksum, uniqueness, age, the face and the ID card photo.
"""

import hashlib
import hmac
import os
import re
import secrets
import time
from collections import deque
from datetime import date

import cv2
import numpy as np

import database as db
import face_ai as ai
import face_model as fm

ID_CARD_DIR = os.path.join(fm.DATA_DIR, "id_cards")
PEPPER_PATH = os.path.join(fm.DATA_DIR, ".id_pepper")

REG_FRAMES = 51
STEADY_FRAMES = 5     # looking straight, eyes open: the liveness reference
CLOSED_FRAMES = 10
REOPEN_FRAMES = 5
MIN_AGE = 18
MAX_ID_CARD_SIDE = 1280
MAX_ATTEMPTS_PER_HOUR = int(os.environ.get("REGISTRATION_LIMIT_PER_HOUR", "30"))  # per device; one shared PC = whole class

GENDERS = {"female": "Female", "male": "Male", "other": "Other", "undisclosed": "Prefer not to say"}
ID_TYPES = {"epic": "Voter ID (EPIC)", "aadhaar": "Aadhaar"}


class RegistrationError(Exception):
    def __init__(self, message, code="invalid", field=None, status=400):
        super().__init__(message)
        self.code, self.field, self.status = code, field, status


# --------------------------------------------------------------------------
# ID numbers
# --------------------------------------------------------------------------

# Verhoeff checksum — the algorithm used for the last digit of an Aadhaar number.
_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_valid(number):
    check = 0
    for i, digit in enumerate(reversed(number)):
        check = _D[check][_P[i % 8][int(digit)]]
    return check == 0


def normalise_id(raw):
    return re.sub(r"[\s-]", "", str(raw or "")).upper()


def validate_id(id_type, number):
    if id_type == "epic":
        if not re.fullmatch(r"[A-Z]{3}[0-9]{7}", number):
            raise RegistrationError("Voter ID must be 3 letters followed by 7 digits, e.g. ABC1234567.",
                                    "id_format", "idNumber")
    elif id_type == "aadhaar":
        if not re.fullmatch(r"[2-9][0-9]{11}", number):
            raise RegistrationError("Aadhaar must be 12 digits and can't start with 0 or 1.", "id_format", "idNumber")
        if not verhoeff_valid(number):
            raise RegistrationError("This Aadhaar number isn't valid (checksum failed). Check for typing mistakes.",
                                    "id_checksum", "idNumber")
    else:
        raise RegistrationError("Choose an ID type.", "id_type", "idType")


def mask_id(id_type, number):
    if id_type == "aadhaar":
        return f"XXXX XXXX {number[-4:]}"
    return f"{number[:3]}****{number[-3:]}"


def _pepper():
    if not os.path.exists(PEPPER_PATH):
        os.makedirs(fm.DATA_DIR, exist_ok=True)
        with open(PEPPER_PATH, "wb") as file:
            file.write(secrets.token_bytes(32))
    with open(PEPPER_PATH, "rb") as file:
        return file.read()


def hash_id(id_type, number):
    return hmac.new(_pepper(), f"{id_type}:{number}".encode(), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------
# Personal details
# --------------------------------------------------------------------------

def age_on(dob, today=None):
    today = today or date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def validate_details(payload):
    name = " ".join(str(payload.get("full_name", "")).split())
    letters = sum(ch.isalpha() for ch in name)
    if not (2 <= len(name) <= 60) or letters < 2 or not all(ch.isalpha() or ch in " .'-" or
                                                           ch.isspace() or 0x0900 <= ord(ch) <= 0x0DFF
                                                           for ch in name):
        raise RegistrationError("Enter your full name (letters only).", "name", "fullName")

    try:
        dob = date.fromisoformat(str(payload.get("dob", "")))
    except ValueError:
        raise RegistrationError("Enter your date of birth.", "dob", "dob")
    age = age_on(dob)
    if age < 0 or age > 120:
        raise RegistrationError("Check your date of birth.", "dob", "dob")
    if age < MIN_AGE:
        raise RegistrationError(f"You must be at least {MIN_AGE} years old to register.", "underage", "dob")

    gender = payload.get("gender")
    if gender not in GENDERS:
        raise RegistrationError("Choose a gender option.", "gender", "gender")

    id_type = payload.get("id_type")
    number = normalise_id(payload.get("id_number"))
    validate_id(id_type, number)

    mobile = re.sub(r"[\s-]", "", str(payload.get("mobile", "") or ""))
    if mobile.startswith("+91"):
        mobile = mobile[3:]
    if mobile and not re.fullmatch(r"[6-9][0-9]{9}", mobile):
        raise RegistrationError("Mobile number must be 10 digits starting with 6-9 (or leave it empty).",
                                "mobile", "mobile")

    if payload.get("consent") is not True:
        raise RegistrationError("Please agree to the consent statement.", "consent", "consent")

    return {
        "full_name": name, "dob": dob.isoformat(), "age": age, "gender": GENDERS[gender],
        "id_type": id_type, "id_number": number, "id_masked": mask_id(id_type, number),
        "id_hash": hash_id(id_type, number), "mobile": mobile,
    }


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def _decode_list(frames, count, what):
    if not isinstance(frames, list) or len(frames) != count:
        raise RegistrationError(f"Camera data is incomplete ({what}). Please capture again.", "frames")
    images = [fm.decode_data_url(frame) for frame in frames]
    if any(image is None for image in images):
        raise RegistrationError("Camera data is damaged. Please capture again.", "frames")
    return images


def _decode_id_card(data_url):
    image = fm.decode_data_url(data_url)
    if image is None:
        raise RegistrationError("Upload a photo of your ID card (JPG or PNG).", "id_card", "idCard")
    height, width = image.shape[:2]
    if min(height, width) < 200:
        raise RegistrationError("The ID card photo is too small. Take a closer, clearer photo.", "id_card", "idCard")
    scale = MAX_ID_CARD_SIDE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def id_card_path(ref):
    return os.path.join(ID_CARD_DIR, f"{ref}.jpg")


def id_card_bytes(ref):
    path = id_card_path(ref)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as file:
        return file.read()


# How close the ID card photo must be to the live face (same distance scale as voting).
# Printed card photos differ more from a webcam image, so this is looser than voting.
ID_CARD_MATCH_THRESHOLD = float(os.environ.get("ID_CARD_MATCH_THRESHOLD", "0.95"))


def card_face_match(card, live_samples):
    """Compare the face printed on the ID card with the live face.

    Returns (status, detail): pass = same person, warn = no face on card or no clear match.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    faces = fm.get_detector().detectMultiScale(gray, 1.1, 4, minSize=(24, 24))
    if not len(faces):
        return "warn", "No face found on the ID card image"
    largest = max(faces, key=lambda f: f[2] * f[3])
    card_feature = fm.to_features([fm.crop_face(card, largest)])[0]
    live_features = fm.to_features(live_samples)
    distance = float(np.median(np.linalg.norm(live_features - card_feature, axis=1)))
    if distance <= ID_CARD_MATCH_THRESHOLD:
        return "pass", f"Card photo matches the live face (distance {distance:.2f})"
    return "warn", f"Card photo doesn't clearly match the live face (distance {distance:.2f}, limit {ID_CARD_MATCH_THRESHOLD})"


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------

_attempts = {}


def rate_limited(ip):
    window = _attempts.setdefault(ip, deque())
    cutoff = time.time() - 3600
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= MAX_ATTEMPTS_PER_HOUR:
        return True
    window.append(time.time())
    return False


# --------------------------------------------------------------------------
# The whole check pipeline
# --------------------------------------------------------------------------

def _check(key, label, status, detail=""):
    return {"key": key, "label": label, "status": status, "detail": detail}


def submit(payload):
    """Validate, verify and store an application. Returns (reference number, status).

    Hard failures raise RegistrationError (nothing is stored). If every check passes and
    auto-approval is on, the voter is approved immediately; otherwise the application waits
    for the election officer with its warnings.
    """
    details = validate_details(payload)
    checks = [
        _check("id_format", "ID number format", "pass",
               f"{ID_TYPES[details['id_type']]} " + ("checksum valid" if details["id_type"] == "aadhaar" else "format valid")),
        _check("age", "Age 18 or over", "pass", f"Age {details['age']}"),
    ]

    existing = db.id_in_use(details["id_hash"])
    if existing:
        raise RegistrationError("This ID number is already registered.", "duplicate_id", "idNumber", 409)
    checks.append(_check("id_unique", "ID not registered before", "pass"))

    id_card = _decode_id_card(payload.get("id_card"))
    frames = _decode_list(payload.get("frames"), REG_FRAMES, "face")

    # Face samples: only frames with exactly one face.
    samples, boxed, crowded = [], [], 0
    for image in frames:
        boxes = fm.detect_faces(image)
        if len(boxes) == 1:
            samples.append(fm.crop_face(image, boxes[0]))
            boxed.append((image, boxes[0]))
        elif len(boxes) > 1:
            crowded += 1
    if len(samples) < fm.MIN_SAMPLES_PER_VOTER:
        raise RegistrationError(
            f"Your face was clear in only {len(samples)} of {REG_FRAMES} frames. "
            "Face the camera in good light, alone, and try again.", "face", None, 422)
    checks.append(_check("face", "Face captured clearly", "pass", f"{len(samples)} of {REG_FRAMES} frames"))
    checks.append(_check("one_person", "Only one person in view",
                         "warn" if crowded > 3 else "pass",
                         f"Another face appeared in {crowded} frames" if crowded else "No other faces"))

    if fm.LIVENESS_REQUIRED:
        steady = _decode_list(payload.get("steady"), STEADY_FRAMES, "looking straight")
        closed = _decode_list(payload.get("closed"), CLOSED_FRAMES, "eyes closed")
        reopened = _decode_list(payload.get("reopened"), REOPEN_FRAMES, "eyes open")
        alive, message, _, metrics = fm.check_liveness(closed, steady, reopened)
        print(f"Registration liveness {'PASS' if alive else 'FAIL'}: {metrics}", flush=True)
        if not alive:
            raise RegistrationError(message + " (liveness check)", "liveness", None, 422)
        checks.append(_check("liveness", "Live person (eyes closed and opened)", "pass"))
    else:
        checks.append(_check("liveness", "Live person check", "skip", "Turned off by the administrator"))

    brightness, sharpness = fm.face_quality(boxed[::5])
    quality_issues = []
    if brightness is not None and brightness < 55:
        quality_issues.append("too dark")
    if brightness is not None and brightness > 215:
        quality_issues.append("too bright")
    if sharpness is not None and sharpness < 25:
        quality_issues.append("blurry")
    checks.append(_check("quality", "Photo quality", "warn" if quality_issues else "pass",
                         ", ".join(quality_issues) or f"brightness {brightness}, sharpness {sharpness}"))

    live_vector = None
    if ai.available():
        # --- AI face recognition (SFace) decides duplicates and the ID card match ---
        ai.backfill(fm._read_dataset)
        live_vector = ai.embed_many([image for image, _ in boxed])
        if live_vector is None:
            raise RegistrationError("The face model couldn't read your face clearly. Face the camera in good light "
                                    "and try again.", "face", None, 422)
        duplicate, score = ai.best_match(live_vector)
        if duplicate and score >= ai.CLEARLY_SAME:
            raise RegistrationError(
                f"This face is already registered (voter {duplicate}). One person can register only once.",
                "duplicate_face", None, 409)
        if duplicate and score >= ai.SAME_PERSON:
            checks.append(_check("duplicate_face", "Face not already registered", "warn",
                                 f"Possibly the same person as {duplicate} (AI similarity {score:.2f})"))
        else:
            checks.append(_check("duplicate_face", "Face not already registered", "pass",
                                 f"No match (closest AI similarity {max(score, 0):.2f})"))
            duplicate = None

        card_vector = ai.embed_largest(id_card, min_score=0.5)
        if card_vector is None:
            raise RegistrationError("We couldn't find a face on your ID card photo. Upload a clear, well-lit photo "
                                    "of the front of the card.", "id_card_face", "idCard", 422)
        card_score = ai.similarity(live_vector, card_vector)
        if card_score < ai.CLEARLY_DIFFERENT:
            raise RegistrationError("The photo on your ID card doesn't match your face. Upload a photo of your own "
                                    "ID card.", "id_card_mismatch", "idCard", 422)
        checks.append(_check("id_card", "ID card photo matches the face",
                             "pass" if card_score >= ai.SAME_PERSON else "warn",
                             f"AI similarity {card_score:.2f} (same person at {ai.SAME_PERSON} or more)"))
    else:
        # Fallback without the AI model files: simpler pixel comparison, warnings only.
        duplicate, percent = fm.find_duplicate(samples)
        checks.append(_check("duplicate_face", "Face not already registered", "warn" if duplicate else "pass",
                             f"Looks like {duplicate} ({percent}% of samples matched)" if duplicate else "No match found"))
        card_status, card_detail = card_face_match(id_card, samples)
        checks.append(_check("id_card", "ID card photo matches the face", card_status, card_detail))

    flags = sum(1 for c in checks if c["status"] == "warn")
    photo = fm.crop_photo(*boxed[min(10, len(boxed) - 1)])
    saved = []

    def store_files(ref):
        saved.append(ref)
        fm.save_pending_samples(ref, samples)
        if live_vector is not None:
            ai.save_pending(ref, live_vector)
        fm.save_photo(ref, photo)
        os.makedirs(ID_CARD_DIR, exist_ok=True)
        cv2.imwrite(id_card_path(ref), id_card, [cv2.IMWRITE_JPEG_QUALITY, 85])  # re-encoded: strips metadata

    # Auto-approval: only when EVERY check passed. Anything suspicious waits for a human.
    auto = approve_files if flags == 0 and db.auto_approve_enabled() else None
    try:
        return db.create_application(details, checks, flags, duplicate, store_files, auto)
    except Exception:
        for ref in saved:
            discard_files(ref, keep_photo=False)
        raise


def approve_files(ref):
    """Move an approved applicant's face samples and AI fingerprint into the voter data."""
    fm.approve_pending_samples(ref)
    ai.approve_pending(ref)


def discard_files(ref, keep_photo=False):
    fm.discard_pending_samples(ref)
    ai.discard_pending(ref)
    for path in [id_card_path(ref)] + ([] if keep_photo else [fm.photo_path(ref)]):
        if os.path.exists(path):
            os.remove(path)
