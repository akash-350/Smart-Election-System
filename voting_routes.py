from flask import Blueprint, current_app, jsonify, render_template, request, session

import database as db
import face_ai as ai
from auth import admin_required
from face_model import (
    LIVENESS_REQUIRED, FaceDatasetError, check_liveness, decode_data_url, identify_frames
)

voting = Blueprint("voting", __name__)

SCAN_FRAMES = 5       # eyes open, for identification
CLOSED_FRAMES = 10    # "close your eyes" (about 2 seconds)
REOPEN_FRAMES = 5     # "open your eyes"


def _clear_booth_session():
    session.pop("pending_demo_voter", None)
    session.pop("confirmed_demo_voter", None)



def _decode_all(frames, count):
    if not isinstance(frames, list) or len(frames) != count:
        return None
    images = [decode_data_url(frame) for frame in frames]
    return None if any(image is None for image in images) else images


def _fail(message, code, status=400, log=True):
    if log:
        db.log_booth_event("Face scan rejected", code)
    return jsonify(error=message, code=code), status


@voting.route("/vote")
@admin_required
def vote_page():
    """The voting booth, run on the officer's screen."""
    election = db.current_election()
    return render_template(
        "vote.html",
        election=election,
        candidates=db.get_candidates(election["election_id"]),
        frames={"scan": SCAN_FRAMES, "closed": CLOSED_FRAMES, "reopen": REOPEN_FRAMES},
        liveness=LIVENESS_REQUIRED,
        nota=db.NOTA,
        remote=False,
    )


@voting.route("/api/identify-demo-voter", methods=["POST"])
@admin_required
def identify_demo_voter():
    _clear_booth_session()
    if db.current_election()["state"] != "open":
        return _fail("Voting is not open.", "not_open", 409, log=False)

    payload = request.get_json(silent=True) or {}
    scan = _decode_all(payload.get("frames"), SCAN_FRAMES)
    closed = reopened = []
    if LIVENESS_REQUIRED:
        closed = _decode_all(payload.get("closed"), CLOSED_FRAMES)
        reopened = _decode_all(payload.get("reopened"), REOPEN_FRAMES)
    if scan is None or closed is None or reopened is None:
        return _fail("Invalid camera data.", "bad_request", log=False)

    try:
        voter_id, message, code = identify_frames(scan + reopened)
        if code in ("no_face", "multiple_faces"):
            return _fail(message, code, 404)
        if ai.available():
            # The AI model (SFace) has the final say on who this is.
            vector = ai.embed_many(scan + reopened)
            ranked = ai.top_two(vector) if vector is not None else []
            confident = (ranked and ranked[0][1] >= ai.SAME_PERSON
                         and (len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= 0.05))
            if confident:
                voter_id = ranked[0][0]
            elif voter_id is not None and ai.get_embedding(voter_id) is not None:
                voter_id = None     # the simple matcher guessed someone the AI doesn't agree with
            if voter_id is None:
                return _fail("Face not recognised as a registered voter.", "not_recognised", 404)
        elif voter_id is None:
            return _fail(message, code, 404)
        if LIVENESS_REQUIRED:
            alive, message, code, metrics = check_liveness(closed, scan, reopened)
            current_app.logger.warning("Booth liveness %s: %s", "PASS" if alive else "FAIL", metrics)
            if not alive:
                return _fail(message, code, 422)
    except FaceDatasetError as error:
        return _fail(str(error), "dataset", log=False)
    except Exception:
        current_app.logger.exception("Face identification failed")
        return _fail("Face matching failed.", "server", 500, log=False)

    if not db.is_registered(voter_id):
        return _fail("This face is not on the voter list.", "not_registered", 404)
    if db.has_voted(voter_id):
        return _fail(f"Voter {voter_id} has already voted.", "already_voted", 409)

    session["pending_demo_voter"] = voter_id
    return jsonify(voter_id=voter_id, name=db.voter_name(voter_id))


@voting.route("/api/confirm-demo-voter", methods=["POST"])
@admin_required
def confirm_demo_voter():
    voter_id = session.get("pending_demo_voter")
    if not voter_id:
        return jsonify(error="Scan your face first.", code="no_voter"), 400
    if db.has_voted(voter_id):
        return jsonify(error="This voter has already voted.", code="already_voted"), 409
    session["confirmed_demo_voter"] = voter_id
    return jsonify(message="Voter confirmed.")


@voting.route("/api/cast-demo-ballot", methods=["POST"])
@admin_required
def cast_demo_ballot():
    voter_id = session.get("confirmed_demo_voter")
    candidate = (request.get_json(silent=True) or {}).get("candidate")
    if not voter_id:
        return jsonify(error="Confirm the voter first.", code="no_voter"), 400
    try:
        receipt = db.record_ballot(voter_id, candidate)
    except db.ElectionError as error:
        return jsonify(error=str(error), code=error.code), 409
    finally:
        _clear_booth_session()
    return jsonify(message="Vote recorded.", receipt=receipt)
