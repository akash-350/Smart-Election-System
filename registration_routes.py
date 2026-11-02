from datetime import date

from flask import Blueprint, Response, abort, current_app, jsonify, render_template, request

import database as db
import face_model as fm
import registration as reg
from auth import admin_required, current_actor

registration = Blueprint("registration", __name__)


# --------------------------------------------------------------------------
# Public: voter self-registration (no login; also used by the officer at the desk)
# --------------------------------------------------------------------------

@registration.route("/register")
def self_register_page():
    return render_template(
        "register_self.html",
        frames={"reg": reg.REG_FRAMES, "steady": reg.STEADY_FRAMES, "closed": reg.CLOSED_FRAMES, "reopen": reg.REOPEN_FRAMES},
        liveness=fm.LIVENESS_REQUIRED,
        genders=reg.GENDERS,
        min_age=reg.MIN_AGE,
        max_dob=date.today().replace(year=date.today().year - reg.MIN_AGE).isoformat(),
    )


@registration.route("/api/self-register", methods=["POST"])
def self_register_api():
    if not request.is_json:
        return jsonify(error="Expected a JSON request."), 415
    if reg.rate_limited(request.remote_addr or "unknown"):
        return jsonify(error="Too many registration attempts from this device. Try again in an hour.",
                       code="rate_limited"), 429
    try:
        ref, status = reg.submit(request.get_json(silent=True) or {})
    except reg.RegistrationError as error:
        if error.code in ("face", "liveness", "duplicate_id", "duplicate_face", "id_card_mismatch", "id_card_face"):
            db.log_booth_event("Registration refused", error.code)
        return jsonify(error=str(error), code=error.code, field=error.field), error.status
    except db.ElectionError as error:
        return jsonify(error=str(error), code=error.code), 409
    except Exception:
        current_app.logger.exception("Self-registration failed")
        return jsonify(error="Something went wrong while saving. Please try again.", code="server"), 500
    return jsonify(ref=ref, status=status)


@registration.route("/register/status", methods=["GET", "POST"])
def status_page():
    result, error, ref = None, None, ""
    if request.method == "POST":
        ref = request.form.get("ref", "").strip().upper()
        dob = request.form.get("dob", "")
        application = db.get_application(ref) if ref else None
        # Require the date of birth too, so reference numbers can't be guessed to see others' status.
        if not application or application["dob"] != dob:
            error = "No application matches that reference number and date of birth."
        else:
            result = application
    return render_template("register_status.html", result=result, error=error, ref=ref)


# --------------------------------------------------------------------------
# Admin: review queue
# --------------------------------------------------------------------------

@registration.route("/registrations")
@admin_required
def registrations_page():
    status = request.args.get("status", "pending")
    if status not in ("pending", "approved", "auto", "rejected", "all"):
        status = "pending"
    applications = db.list_applications(None if status == "all" else status)
    today = date.today()
    for application in applications:
        application["age"] = reg.age_on(date.fromisoformat(application["dob"]), today)
    return render_template("registrations.html", applications=applications, status=status,
                           counts=db.application_counts(), auto_approve=db.auto_approve_enabled(),
                           auto_approver=db.AUTO_APPROVER)


@registration.route("/api/settings/auto-approve", methods=["POST"])
@admin_required
def set_auto_approve():
    enabled = bool((request.get_json(silent=True) or {}).get("enabled"))
    db.set_setting("auto_approve", "on" if enabled else "off", current_actor())
    return jsonify(ok=True, enabled=enabled)


@registration.route("/registrations/<ref>")
@admin_required
def registration_detail(ref):
    application = db.get_application(ref) or abort(404)
    application["age"] = reg.age_on(date.fromisoformat(application["dob"]))
    duplicate_name = db.voter_name(application["duplicate_of"]) if application["duplicate_of"] else None
    return render_template("registration_detail.html", a=application, id_types=reg.ID_TYPES,
                           auto_approver=db.AUTO_APPROVER,
                           duplicate_name=duplicate_name,
                           has_id_card=reg.id_card_bytes(ref) is not None)


@registration.route("/registrations/<ref>/photo")
@admin_required
def registration_photo(ref):
    data = fm.photo_bytes(ref) if db.get_application(ref) or db.is_registered(ref) else None
    if data is None:
        abort(404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, no-store"})


@registration.route("/registrations/<ref>/id-card")
@admin_required
def registration_id_card(ref):
    data = reg.id_card_bytes(ref) if db.get_application(ref) else None
    if data is None:
        abort(404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, no-store"})


@registration.route("/api/registrations/<ref>/approve", methods=["POST"])
@admin_required
def approve(ref):
    try:
        db.approve_application(ref, current_actor(), reg.approve_files)
    except fm.FaceDatasetError as error:
        return jsonify(error=str(error)), 409
    return jsonify(ok=True)


@registration.route("/api/registrations/<ref>/reject", methods=["POST"])
@admin_required
def reject(ref):
    reason = (request.get_json(silent=True) or {}).get("reason")
    db.reject_application(ref, reason, current_actor())
    # Data minimisation: a rejected applicant's face and ID card images are deleted.
    reg.discard_files(ref)
    return jsonify(ok=True)
