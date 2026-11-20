# Netflix Content Analyzer - containerised EDA pipeline
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
RUN useradd -m runner && mkdir -p data outputs && chown -R runner /app
USER runner

# Mount ./outputs to retrieve the dashboards:  docker run -v $(pwd)/outputs:/app/outputs netflix-analyzer
VOLUME ["/app/outputs"]
ENTRYPOINT ["python", "pipeline.py"]
