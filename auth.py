import hmac
import os
import secrets
import time
from functools import wraps

from flask import jsonify, redirect, request, session, url_for

ADMIN_USERNAME = os.environ.get("DEMO_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("DEMO_ADMIN_PASSWORD", "demo123")
USING_DEFAULT_PASSWORD = "DEMO_ADMIN_PASSWORD" not in os.environ

# A random key is generated when none is configured, so the session cookie
# can't be forged with a known key. (Admins must log in again after a restart.)
SECRET_KEY = os.environ.get("DEMO_SECRET_KEY") or secrets.token_hex(32)


MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60
_failed_logins = {}  # ip -> (count, first_failure_time)


def login_locked(ip):
    count, since = _failed_logins.get(ip, (0, 0))
    if count >= MAX_FAILED_LOGINS and time.time() - since < LOCKOUT_SECONDS:
        return True
    if count and time.time() - since >= LOCKOUT_SECONDS:
        _failed_logins.pop(ip, None)
    return False


def check_credentials(username, password, ip):
    user_ok = hmac.compare_digest(username.encode(), ADMIN_USERNAME.encode())
    pass_ok = hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode())
    if user_ok and pass_ok:
        _failed_logins.pop(ip, None)
        return True
    count, since = _failed_logins.get(ip, (0, time.time()))
    _failed_logins[ip] = (count + 1, since)
    return False


def admin_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            if request.path.startswith("/api/"):
                return jsonify(error="Admin login required.", code="login"), 401
            return redirect(url_for("login"))
        # State-changing calls must be JSON. Browsers can't send cross-site
        # JSON without a CORS preflight, so this blocks CSRF via hidden forms.
        if request.method == "POST" and not request.is_json:
            return jsonify(error="Expected a JSON request."), 415
        return function(*args, **kwargs)
    return wrapper


def current_actor():
    return session.get("admin_user", ADMIN_USERNAME)
