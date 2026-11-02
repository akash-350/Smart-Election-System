"""Production server for Smart Election.

    python serve.py              # serves http://127.0.0.1:5000 with Waitress

On first run it creates a private .env file with a strong admin password and a secret key
(printed once). Put it online with a tunnel, e.g. start_live.bat (Cloudflare Tunnel).
"""
import os
import secrets

# On a cloud host, point this at the persistent disk (e.g. data/.env) so secrets survive restarts.
ENV_PATH = os.environ.get("SMART_ENV_FILE", ".env")


def load_env():
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def ensure_secrets():
    new = {}
    if not os.environ.get("DEMO_SECRET_KEY"):
        new["DEMO_SECRET_KEY"] = secrets.token_hex(32)
    if not os.environ.get("DEMO_ADMIN_PASSWORD"):
        new["DEMO_ADMIN_PASSWORD"] = secrets.token_urlsafe(12)
    if new:
        with open(ENV_PATH, "a", encoding="utf-8") as file:
            file.write("# Smart Election private settings - never share or commit this file\n")
            for key, value in new.items():
                file.write(f"{key}={value}\n")
        os.environ.update(new)
        if "DEMO_ADMIN_PASSWORD" in new:
            print("=" * 62)
            print(" New admin password created (saved in .env):")
            print(f"   username: {os.environ.get('DEMO_ADMIN_USERNAME', 'admin')}")
            print(f"   password: {new['DEMO_ADMIN_PASSWORD']}")
            print("=" * 62)


class RealClientIP:
    """Behind Cloudflare Tunnel every request arrives from 127.0.0.1. Use the visitor's real
    address (set by Cloudflare) so login lockouts and rate limits apply per person, not to
    everyone at once. Only trusted for requests that come from this machine (the tunnel)."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if environ.get("REMOTE_ADDR") in ("127.0.0.1", "::1"):
            real = environ.get("HTTP_CF_CONNECTING_IP") or \
                environ.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip()
            if real:
                environ["REMOTE_ADDR"] = real
            if environ.get("HTTP_X_FORWARDED_PROTO") == "https":
                environ["wsgi.url_scheme"] = "https"
        return self.app(environ, start_response)


if __name__ == "__main__":
    load_env()
    ensure_secrets()

    from waitress import serve  # noqa: E402
    from app import app  # noqa: E402  (imported after the environment is ready)

    port = int(os.environ.get("PORT", "5000"))
    print(f"Smart Election running on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    # The tunnel (Tailscale Funnel or Cloudflare) connects from this PC and adds the visitor's
    # address to X-Forwarded-For. Trust only that local hop, and only the last address it added,
    # so a visitor can't fake their address. (Waitress drops these headers unless told to trust them.)
    serve(RealClientIP(app), host="127.0.0.1", port=port, threads=8,
          max_request_body_size=20 * 1024 * 1024, channel_timeout=120,
          trusted_proxy="127.0.0.1", trusted_proxy_count=1,
          trusted_proxy_headers={"x-forwarded-for", "x-forwarded-proto", "x-forwarded-host"})
