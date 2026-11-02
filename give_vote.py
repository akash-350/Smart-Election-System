"""Desktop voting booth (OpenCV window). The web booth at /vote does the same job."""

import threading
import time
from collections import deque

import cv2
import numpy as np

import database as db
from face_model import (
    EYE_NOISE_FLOOR, LIVENESS_MIN_CHANGE, LIVENESS_REQUIRED, FaceDatasetError, crop_face, detect_faces,
    eye_band, identify, load_model
)

WINDOW = "Smart Election - Voting Booth"
STABLE_FRAMES = 10        # consecutive frames that must agree on the voter
CLOSED_EYE_FRAMES = 3     # liveness: frames with eyes closed
OPEN_EYE_FRAMES = 3       # liveness: frames with eyes open again
LIVENESS_TIMEOUT = 10     # seconds to complete the eye check

# Civic-light colours (BGR)
BG = (250, 248, 247)
NAVY = (138, 58, 30)
GREEN = (61, 128, 21)
SAFFRON = (11, 158, 245)
TEXT = (42, 23, 15)
MUTED = (139, 116, 100)
WHITE = (255, 255, 255)
RED = (28, 28, 185)
PANEL = 320


def beep(times=1, frequency=880):
    """One beep = close your eyes, two beeps = open them (Windows only; silent elsewhere)."""
    def run():
        try:
            import winsound
            for _ in range(times):
                winsound.Beep(frequency, 180)
                time.sleep(0.08)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def put(canvas, text, org, scale=0.6, colour=TEXT, thickness=1):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def main():
    db.initialize_database()
    election = db.current_election()
    if election["state"] != "open":
        print(f"Voting is not open for '{election['name']}' (state: {election['state']}).")
        print("Open voting from the web admin: Election page -> Open voting.")
        return

    candidates = db.get_candidates(election["election_id"])[:9]
    keys = {str(i): name for i, name in enumerate(candidates, start=1)}

    try:
        load_model()
    except (FaceDatasetError, RuntimeError) as error:
        print(error)
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Webcam could not be opened.")
        return

    state = {
        "voter": None,          # voter allowed to vote right now
        "candidate_id": None,   # voter currently being recognised
        "streak": 0,
        "phase": "recognise",   # recognise -> close_eyes -> open_eyes -> ready
        "eye_count": 0,
        "phase_started": 0.0,
        "open_bands": deque(maxlen=8),   # eye area while eyes are open (the reference)
        "reference": None,
        "noise": EYE_NOISE_FLOOR,
        "closed_level": 0.0,
        "frame_width": 640,
        "message": "Look at the camera",
        "done": False,
    }

    def reset(message):
        state.update(voter=None, candidate_id=None, streak=0, phase="recognise", eye_count=0,
                     closed_level=0.0, message=message)
        state["open_bands"].clear()

    def row_geometry(index):
        top = 90 + index * 52
        return top, top + 42

    def select_candidate(key):
        if state["voter"] is None or state["phase"] != "ready":
            state["message"] = "Finish the face check first"
            return
        try:
            receipt = db.record_ballot(state["voter"], keys[key])
        except db.ElectionError as error:
            reset(str(error))
            return
        state["message"] = "Thank you! Your vote has been recorded."
        state["done"] = True
        print("Vote recorded (secret ballot). Receipt code:", receipt, "- check it at /verify")

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        left = state["frame_width"] + 16
        if left <= x <= left + PANEL - 32:
            for index, key in enumerate(keys):
                top, bottom = row_geometry(index)
                if top <= y <= bottom:
                    select_candidate(key)
                    break

    try:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

        while True:
            success, frame = camera.read()
            if not success:
                print("Camera frame error.")
                break

            frame_width = frame.shape[1]
            state["frame_width"] = frame_width
            boxes = detect_faces(frame)

            if len(boxes) != 1:
                if state["phase"] != "recognise" or state["streak"]:
                    reset("Only one face should be visible" if len(boxes) > 1 else "Look at the camera")
                else:
                    state["message"] = "Only one face should be visible" if len(boxes) > 1 else "Look at the camera"
            else:
                box = boxes[0]
                x, y, w, h = box

                if state["phase"] == "recognise":
                    voter, reason = identify(crop_face(frame, box))
                    if voter is not None and voter == state["candidate_id"]:
                        state["streak"] += 1
                    else:
                        state.update(candidate_id=voter, streak=1 if voter else 0)
                        state["open_bands"].clear()
                    if voter is not None:
                        state["open_bands"].append(eye_band(frame, box))

                    if voter is None:
                        state["message"] = reason
                    elif state["streak"] < STABLE_FRAMES:
                        state["message"] = "Recognising... hold still"
                    elif not db.is_registered(voter):
                        reset("This face is not on the voter list")
                    elif db.has_voted(voter):
                        reset(f"{voter} has already voted")
                    elif LIVENESS_REQUIRED:
                        bands = list(state["open_bands"])
                        reference = np.median(bands, axis=0)
                        noise = np.median([np.mean(np.abs(b - reference)) for b in bands])
                        state.update(phase="close_eyes", eye_count=0, phase_started=time.time(),
                                     reference=reference, noise=max(float(noise), EYE_NOISE_FLOOR),
                                     message="1 beep: CLOSE your eyes")
                        beep(1, 660)
                    else:
                        state.update(voter=voter, phase="ready", message="Choose a candidate")

                elif state["phase"] in ("close_eyes", "open_eyes"):
                    if time.time() - state["phase_started"] > LIVENESS_TIMEOUT:
                        reset("Eye check timed out. Try again.")
                    else:
                        # How different the eye area looks from the eyes-open reference.
                        change = float(np.mean(np.abs(eye_band(frame, box) - state["reference"]))) / state["noise"]
                        if state["phase"] == "close_eyes":
                            closed = change >= LIVENESS_MIN_CHANGE
                            state["eye_count"] = state["eye_count"] + 1 if closed else 0
                            state["closed_level"] = max(state["closed_level"], change) if closed else state["closed_level"]
                            if state["eye_count"] >= CLOSED_EYE_FRAMES:
                                state.update(phase="open_eyes", eye_count=0, phase_started=time.time(),
                                             message="2 beeps: OPEN your eyes")
                                beep(2, 990)
                        else:
                            reopened = change <= max(state["closed_level"] * 0.75, LIVENESS_MIN_CHANGE * 0.9)
                            state["eye_count"] = state["eye_count"] + 1 if reopened else 0
                            if state["eye_count"] >= OPEN_EYE_FRAMES:
                                # Make sure it's still the same person after the eye check.
                                voter, _ = identify(crop_face(frame, box))
                                if voter == state["candidate_id"]:
                                    state.update(voter=voter, phase="ready", message="Choose a candidate")
                                else:
                                    reset("Face changed during the check. Try again.")

                colour = GREEN if state["phase"] == "ready" else SAFFRON if state["candidate_id"] else RED
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 3)

            # --- draw ---
            canvas = cv2.copyMakeBorder(frame, 0, 0, 0, PANEL, cv2.BORDER_CONSTANT, value=BG)
            cv2.rectangle(canvas, (0, 0), (frame_width, 48), WHITE, -1)
            put(canvas, state["message"], (14, 31), 0.65, NAVY, 2)
            if state["voter"]:
                put(canvas, f"Voter: {state['voter']}", (14, frame.shape[0] - 16), 0.6, WHITE, 2)

            left = frame_width + 16
            put(canvas, election["name"][:22], (left, 36), 0.7, TEXT, 2)
            put(canvas, "Choose one option", (left, 66), 0.5, MUTED)
            for index, (key, name) in enumerate(keys.items()):
                top, bottom = row_geometry(index)
                ready = state["phase"] == "ready"
                cv2.rectangle(canvas, (left, top), (left + PANEL - 32, bottom), NAVY if ready else (225, 220, 215), -1)
                put(canvas, f"{key}   {name}", (left + 14, top + 28), 0.65, WHITE if ready else MUTED, 2)

            put(canvas, f"Click or press 1-{len(keys)}   |   Q to quit", (left, frame.shape[0] - 18), 0.45, MUTED)
            cv2.imshow(WINDOW, canvas)

            if state["done"]:
                cv2.waitKey(2000)
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if chr(key) in keys:
                select_candidate(chr(key))
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
