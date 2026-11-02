FROM python:3.12-slim

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .
RUN python download_models.py

# data/ holds the votes database, face data and secrets: mount a persistent disk at /app/data
ENV SMART_ENV_FILE=data/.env \
    PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "mkdir -p data && exec waitress-serve --host=0.0.0.0 --port=${PORT} --threads=8 --channel-timeout=120 --max-request-body-size=20971520 wsgi:application"]
