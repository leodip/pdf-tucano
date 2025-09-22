FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps for pdf2image (poppler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pdf_tucano ./pdf_tucano
COPY main.py ./

ENV STORAGE_ROOT=/data
VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "main.py"]
