"""WSGI entry point for cloud hosting (PythonAnywhere, Render, Railway, ...).

The host's web server imports `application` from this file. Runs 24/7 without this PC.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)              # the app uses paths like data/ and models/ relative to here
sys.path.insert(0, BASE_DIR)

from serve import ensure_secrets, load_env  # noqa: E402

load_env()
ensure_secrets()

import cv2  # noqa: E402

# Cloud hosts give the app one CPU core but OpenCV sees all of the machine's cores and starts a
# thread per core; they fight over the one core and face checks become ~25x slower. Use one.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402

from app import app  # noqa: E402

app.config["SESSION_COOKIE_SECURE"] = True   # cloud hosts always serve over HTTPS
# The host's load balancer sits in front of the app and adds the visitor's real IP/HTTPS
# headers. Trust exactly one hop so login lockouts apply per visitor.
application = ProxyFix(app, x_for=1, x_proto=1, x_host=1)
