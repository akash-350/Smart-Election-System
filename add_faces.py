import re

import cv2

from database import initialize_database, is_registered, register_voter
from face_model import (
    FaceDatasetError, MIN_SAMPLES_PER_VOTER, crop_face, crop_photo, detect_faces,
    registered_ids, save_photo, save_voter_samples
)

TOTAL_FRAMES = 51
CAPTURE_INTERVAL = 2


def main():
    initialize_database()

    voter_id = input("Enter demo voter ID: ").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,30}", voter_id):
        print("Voter ID must be 1-30 letters, digits, '_' or '-'.")
        return

    try:
        if voter_id in registered_ids() or is_registered(voter_id):
            print("Demo voter already registered.")
            return
    except FaceDatasetError as error:
        print(error)
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Webcam could not be opened.")
        return

    face_samples = []
    photo = None
    frame_count = 0
    print("Look at camera. Press Q to cancel.")

    try:
        while len(face_samples) < TOTAL_FRAMES:
            success, frame = camera.read()
            if not success:
                print("Camera frame error.")
                break

            boxes = detect_faces(frame)
            # Only sample when exactly one face is visible, so nobody else's
            # face ends up in this voter's samples.
            if len(boxes) == 1:
                if frame_count % CAPTURE_INTERVAL == 0:
                    face_samples.append(crop_face(frame, boxes[0]))
                    if photo is None and len(face_samples) >= 10:
                        photo = crop_photo(frame, boxes[0])
                frame_count += 1

            for (x, y, w, h) in boxes:
                colour = (0, 255, 0) if len(boxes) == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

            status = f"Samples: {len(face_samples)}/{TOTAL_FRAMES}"
            if len(boxes) > 1:
                status += "  (only one face please)"
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Demo Voter Registration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(face_samples) < MIN_SAMPLES_PER_VOTER:
        print("Registration cancelled. No data saved.")
        return

    try:
        save_voter_samples(voter_id, face_samples)
    except (ValueError, FaceDatasetError) as error:
        print(error)
        return

    if photo is not None:
        save_photo(voter_id, photo)
    register_voter(voter_id, "desktop")
    print(f"Demo voter registered successfully with {len(face_samples)} samples!")


if __name__ == "__main__":
    main()
