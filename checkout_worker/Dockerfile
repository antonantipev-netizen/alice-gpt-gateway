FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py .

ENV PYTHONUNBUFFERED=1
ENV BROWSER_DATA_DIR=/data/browser
ENV SCREENSHOT_PATH=/data/last_checkout.png
ENV HEADLESS=1

VOLUME ["/data"]

CMD ["python", "worker.py"]
