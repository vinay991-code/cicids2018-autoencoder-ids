# Dashboard + scoring API (CPU only, numpy inference: torch is not installed). Build from the project folder:  docker build -t ids-dashboard .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv/ids
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt
# code, saved models, measured results and the scaler. The large data files are NOT copied (see docker-compose.yml for the replay mount).
COPY app ./app
COPY run_app.py .
COPY models ./models
COPY results ./results
COPY results_trim ./results_trim
COPY seeds/seeds_summary.json ./seeds/seeds_summary.json
COPY data/processed/scaler.json ./data/processed/scaler.json
RUN useradd --create-home ids && chown -R ids /srv/ids
USER ids
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
# inside the container the server listens on all interfaces; docker-compose publishes it on 127.0.0.1 only
CMD ["python", "run_app.py", "--host", "0.0.0.0", "--port", "8000"]
