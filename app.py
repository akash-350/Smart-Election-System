import csv
import gzip
import os
import re
from io import StringIO

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, session, url_for

import database as db
import face_ai as ai
import face_model as fm
from auth import (
    SECRET_KEY, USING_DEFAULT_PASSWORD, admin_required, check_credentials, current_actor, login_locked
)
from face_model import (
    LIVENESS_REQUIRED, MATCH_THRESHOLD, FaceDatasetError, crop_face, crop_photo, decode_data_url,
    detect_faces, evaluate, find_duplicate, photo_bytes, registered_ids, remove_voter_samples,
    save_photo, save_voter_samples,
)

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SEND_FILE_MAX_AGE_DEFAULT=7 * 24 * 3600,   # static files are cached; asset() adds ?v= so updates still load
)


@app.template_global()
def asset(filename):
    """URL for a static file with a version stamp, so browsers can cache it safely."""
    try:
        version = int(os.path.getmtime(os.path.join(app.static_folder, filename)))
    except OSError:
        version = 0
    return url_for("static", filename=filename, v=version)


COMPRESSIBLE = ("text/", "application/json", "application/javascript", "image/svg+xml")


@app.after_request
def compress(response):
    """Gzip text responses: pages load much faster over slow mobile networks and tunnels."""
    if (response.status_code < 200 or response.status_code >= 300 or response.direct_passthrough
            or "gzip" not in request.headers.get("Accept-Encoding", "")
            or "Content-Encoding" in response.headers
            or not (response.mimetype or "").startswith(COMPRESSIBLE)):
        return response
    data = response.get_data()
    if len(data) < 1024:
        return response
    response.set_data(gzip.compress(data, compresslevel=6))
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Vary"] = "Accept-Encoding"
    return response

VOTER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,30}$")
REGISTRATION_FRAMES = 51

db.initialize_database()
if ai.available():
    ai.backfill(fm._read_dataset)   # AI fingerprints for voters registered before the model was added
else:
    print("NOTE: AI face models not found in models/. Run: python download_models.py")


# Every page and API needs the admin login, except voter self-registration and its status
# check. Voting itself happens only at the officer's booth.
PUBLIC_ENDPOINTS = {"login", "logout", "static", "registration.self_register_page",
                    "registration.self_register_api", "registration.status_page"}


@app.before_request
def require_admin():
    if request.endpoint in PUBLIC_ENDPOINTS or session.get("admin_logged_in"):
        return None
    if request.path.startswith("/api/"):
        return jsonify(error="Admin login required.", code="login"), 401
    return redirect(url_for("login"))


@app.context_processor
def inject_globals():
    if not session.get("admin_logged_in"):
        return {"current": None, "pending_count": 0}
    return {"current": db.current_election(), "pending_count": db.application_counts()["pending"]}


@app.errorhandler(db.ElectionError)
def election_error(error):
    return jsonify(error=str(error), code=error.code), 409


def json_payload():
    return request.get_json(silent=True) or {}


# --------------------------------------------------------------------------
# Login
# --------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        if login_locked(ip):
            return render_template("login.html", error="Too many failed attempts. Try again in a minute."), 429
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if check_credentials(username, password, ip):
            session.clear()
            session["admin_logged_in"] = True
            session["admin_user"] = username
            with db.transaction() as connection:
                db.log_event(connection, username, "Admin signed in", f"from {ip}")
            return redirect(url_for("dashboard"))
        with db.transaction() as connection:
            db.log_event(connection, "unknown", "Failed sign-in", f"from {ip}")
        return render_template("login.html", error="Invalid username or password.", username=username), 401
    if session.get("admin_logged_in"):
        return redirect(url_for("dashboard"))
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

@app.route("/")
@admin_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/results")
@admin_required
def results_api():
    election = db.current_election()
    data = {"election": election, "turnout": db.turnout_summary(election),
            "candidates": db.get_candidates(election["election_id"])}
    # Results stay sealed until voting closes — not even the admin can peek.
    if election["state"] == "closed":
        data["results"] = db.election_results(election)
    return jsonify(data)


# --------------------------------------------------------------------------
# Election & candidates
# --------------------------------------------------------------------------

@app.route("/election")
@admin_required
def election_page():
    election = db.current_election()
    return render_template(
        "election.html",
        election=election,
        candidates=db.get_candidates(election["election_id"]),
        turnout=db.turnout_summary(election),
        history=db.list_elections(),
        nota=db.NOTA,
    )


@app.route("/api/election/<action>", methods=["POST"])
@admin_required
def election_action(action):
    payload = json_payload()
    actor = current_actor()
    if action == "rename":
        db.rename_election(payload.get("name"), actor)
    elif action == "open":
        db.open_election(actor)
    elif action == "close":
        election = db.current_election()
        if payload.get("confirm") != election["name"]:
            return jsonify(error="Type the election name exactly to confirm."), 400
        db.close_election(actor)
    elif action == "new":
        db.new_election(payload.get("name"), actor)
    else:
        abort(404)
    return jsonify(ok=True, election=db.current_election())


@app.route("/api/stats")
@admin_required
def stats_api():
    election = db.current_election()
    data = db.registration_stats()
    data["timeline"] = db.turnout_timeline(election) if election["state"] != "setup" else []
    return jsonify(data)


@app.route("/live")
@admin_required
def live_screen():
    """Full-screen projector view: live turnout while open, results when closed."""
    return render_template("live.html")


@app.route("/api/live")
@admin_required
def live_api():
    election = db.current_election()
    data = {"election": {k: election[k] for k in ("name", "state", "opened_at", "closed_at")},
            "turnout": db.turnout_summary(election),
            "timeline": db.turnout_timeline(election) if election["state"] != "setup" else []}
    if election["state"] == "closed":
        data["results"] = db.election_results(election)
    return jsonify(data)


@app.route("/verify", methods=["GET", "POST"])
@admin_required
def verify_vote():
    """The officer checks a voter's receipt code. Says only whether the vote was counted."""
    result, code = None, ""
    if request.method == "POST":
        code = request.form.get("code", "")
        result = db.verify_receipt(code) or False
    return render_template("verify.html", result=result, code=code)


@app.route("/api/candidates/<action>", methods=["POST"])
@admin_required
def candidate_action(action):
    payload = json_payload()
    actor = current_actor()
    if action == "add":
        db.add_candidate(payload.get("name"), actor)
    elif action == "rename":
        db.rename_candidate(payload.get("old"), payload.get("name"), actor)
    elif action == "remove":
        db.remove_candidate(payload.get("name"), actor)
    else:
        abort(404)
    return jsonify(ok=True, candidates=db.get_candidates())


# --------------------------------------------------------------------------
# Results declaration
# --------------------------------------------------------------------------

@app.route("/results")
@app.route("/results/<int:election_id>")
@admin_required
def results_page(election_id=None):
    # Results are only shown for closed elections (open ones show "sealed").
    if election_id is None:
        closed = [e for e in db.list_elections() if e["state"] == "closed"]
        current = db.current_election()
        election = current if current["state"] == "closed" or not closed else closed[0]
    else:
        election = db.get_election(election_id) or abort(404)
    results = db.election_results(election) if election["state"] == "closed" else None
    return render_template(
        "results.html",
        election=election,
        results=results,
        turnout=db.turnout_summary(election),
        history=db.list_elections(),
        nota=db.NOTA,
    )


@app.route("/export/csv")
@admin_required
def export_csv():
    election_id = request.args.get("election", type=int)
    election = db.get_election(election_id) if election_id else db.current_election()
    if not election:
        abort(404)
    if election["state"] != "closed":
        return render_template("message.html", title="Results are sealed",
                               message="Results can be exported after voting closes."), 403

    results = db.election_results(election)
    turnout = db.turnout_summary(election)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Election", election["name"]])
    writer.writerow(["Opened", election["opened_at"], "Closed", election["closed_at"]])
    writer.writerow(["Turnout", f"{turnout['voted']} of {turnout['registered']} ({turnout['participation']}%)"])
    writer.writerow(["Ballot chain", results["checks"]["chain_message"]])
    writer.writerow([])
    writer.writerow(["Candidate", "Votes", "Vote Percentage"])
    for name, votes, pct in zip(results["labels"], results["values"], results["percentages"]):
        writer.writerow([name, votes, pct])
    writer.writerow(["TOTAL", results["total"], "100%" if results["total"] else "0%"])
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", election["name"]).strip("_") or "election"
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}_results.csv"})


# --------------------------------------------------------------------------
# Voters
# --------------------------------------------------------------------------

@app.route("/voters")
@admin_required
def voters():
    rows = db.list_voters()
    return render_template("voters.html", rows=rows, voted_count=sum(r["voted_now"] for r in rows))


@app.route("/voters/<voter_id>/photo")
@admin_required
def voter_photo(voter_id):
    if not VOTER_ID_PATTERN.match(voter_id):
        abort(404)
    data = photo_bytes(voter_id)
    if data is None:
        abort(404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=60"})


@app.route("/api/voters/<voter_id>/delete", methods=["POST"])
@admin_required
def delete_voter(voter_id):
    db.delete_voter(voter_id, current_actor())  # refused by the database if they ever voted
    remove_voter_samples(voter_id)
    ai.remove_embedding(voter_id)
    return jsonify(ok=True)


@app.route("/register-voter")
@admin_required
def register_voter():
    retake = request.args.get("retake", "")
    if retake and not db.is_registered(retake):
        retake = ""
    return render_template("register.html", frame_count=REGISTRATION_FRAMES, retake=retake)


@app.route("/api/register-face", methods=["POST"])
@admin_required
def register_face():
    payload = json_payload()
    voter_id = str(payload.get("voter_id", "")).strip()
    full_name = " ".join(str(payload.get("full_name", "") or "").split())[:60] or None
    frames = payload.get("frames", [])
    retake = bool(payload.get("retake"))

    if not VOTER_ID_PATTERN.match(voter_id):
        return jsonify(error="Invalid demo voter ID."), 400
    if not isinstance(frames, list) or len(frames) != REGISTRATION_FRAMES:
        return jsonify(error=f"Exactly {REGISTRATION_FRAMES} camera frames are required."), 400

    exists = db.is_registered(voter_id)
    if retake and not exists:
        return jsonify(error="Voter not found."), 404
    if not retake and exists:
        return jsonify(error="Demo voter ID already registered."), 409

    face_samples, photo, face_images = [], None, []
    for item in frames:
        image = decode_data_url(item)
        if image is None:
            return jsonify(error="Invalid camera frame received."), 400
        boxes = detect_faces(image)
        if len(boxes) == 1:
            face_images.append(image)
            face_samples.append(crop_face(image, boxes[0]))
            if photo is None and len(face_samples) >= 10:
                photo = crop_photo(image, boxes[0])

    try:
        if not retake and voter_id in registered_ids():
            return jsonify(error="Voter ID already exists in face dataset."), 409
        vector = ai.embed_many(face_images) if ai.available() and face_images else None
        if vector is not None:
            duplicate, score = ai.best_match(vector, exclude=voter_id)
            if duplicate and score >= ai.SAME_PERSON:
                return jsonify(error=f"This face is already registered as {duplicate} (AI similarity {score:.2f}). "
                                     "One person can only register once.", code="duplicate_face",
                               duplicate=duplicate), 409
        elif len(face_samples) >= 5:
            duplicate, percent = find_duplicate(face_samples, exclude=voter_id)
            if duplicate and duplicate != voter_id:
                return jsonify(error=f"This face looks like already-registered {duplicate} "
                                     f"({percent}% of samples matched). One person can only register once.",
                               code="duplicate_face", duplicate=duplicate), 409
        save_voter_samples(voter_id, face_samples, replace=retake)
        if vector is not None:
            ai.set_embedding(voter_id, vector)
        if photo is not None:
            save_photo(voter_id, photo)
        if retake:
            db.log_face_update(voter_id, current_actor())
        else:
            db.register_voter(voter_id, current_actor(), full_name)
    except ValueError as error:
        return jsonify(error=f"{error} Improve lighting and try again."), 400
    except FaceDatasetError as error:
        return jsonify(error=str(error)), 409
    except Exception:
        app.logger.exception("Registration failed")
        return jsonify(error="Could not save voter registration."), 500

    verb = "updated" if retake else "registered"
    return jsonify(message=f"Voter {voter_id} {verb} with {len(face_samples)} face samples.")


# --------------------------------------------------------------------------
# Audit & accuracy
# --------------------------------------------------------------------------

@app.route("/audit")
@admin_required
def audit():
    return render_template(
        "audit.html",
        entries=db.audit_entries(),
        audit_check=db.verify_audit_chain(),
        ballot_check=db.verify_ballot_chain(),
    )


@app.route("/accuracy")
@admin_required
def accuracy():
    report, error = None, None
    try:
        report = evaluate()
    except FaceDatasetError as exc:
        error = str(exc)
    return render_template("accuracy.html", report=report, error=error,
                           threshold=MATCH_THRESHOLD, liveness=LIVENESS_REQUIRED)


from voting_routes import voting  # noqa: E402  (blueprint uses the app's helpers)
from registration_routes import registration  # noqa: E402
app.register_blueprint(voting)
app.register_blueprint(registration)

if __name__ == "__main__":
    if USING_DEFAULT_PASSWORD:
        print("WARNING: using the default admin password 'demo123'. "
              "Set DEMO_ADMIN_PASSWORD to change it.")
    app.run(host="127.0.0.1", port=5000, debug=False)
