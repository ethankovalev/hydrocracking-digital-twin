FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.py check_setup.py run_all.py ./
COPY src/ ./src

VOLUME ["/app/data", "/app/outputs"]

CMD ["python", "check_setup.py"]
