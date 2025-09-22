# pdf-tucano

A standalone PDF-to-Markdown microservice for converting PDF documents into Markdown. Upload a PDF, poll for status, and retrieve Markdown output once conversion completes. Jobs are persisted in PostgreSQL and PDFs are stored on disk, while conversion relies on OpenRouter for OCR/LLM transcription.

## Features

- Asynchronous job queue with status and result endpoints
- Conversion workflow: pdf2image → PNG → OpenRouter → Markdown
- Configurable DPI, scaling width, and concurrency via environment variables
- PostgreSQL persistence with per-page tracking and filesystem PDF storage
- Lightweight admin UI at `/admin` for monitoring job status and results
- Container image (`pdf-tucano`) ready for deployment

## Quickstart

1. **Set environment variables**

   Copy `.env.example` to `.env` and fill in your OpenRouter key, or export values manually:

```bash
OPENROUTER_API_KEY=your-openrouter-api-key
```

2. **Run with Docker Compose (example)**

```yaml
services:
  postgres:
    image: postgres:17
    restart: unless-stopped
    environment:
      POSTGRES_DB: pdf_tucano
      POSTGRES_USER: postgresuser
      POSTGRES_PASSWORD: postgresPASS123
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgresuser -d pdf_tucano"]
      interval: 5s
      timeout: 5s
      retries: 5
    volumes:
      - pgdata:/var/lib/postgresql/data

  pdf-tucano:
    build: .
    restart: unless-stopped    
    environment:
      DATABASE_URL: postgresql+psycopg2://postgresuser:postgresPASS123@postgres:5432/pdf_tucano
      OPENROUTER_MODEL: google/gemini-2.0-flash-001
      PAGE_IMAGE_WIDTH: 1500
      PAGE_IMAGE_DPI: 300
      MAX_CONCURRENT_PAGES: 8
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
      LOG_LEVEL: DEBUG
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "8000:8000"
    volumes:
      - pdfdata:/data

volumes:
  pgdata:
  pdfdata:
```

> **Note:** The PostgreSQL service is only reachable from other containers on the Compose network (e.g., via hostname `postgres`).
If you previously created the `pgdata` volume with a different Postgres major version, remove it before starting (`docker volume rm pdf-tucano_pgdata`) or Docker will report an incompatible data-directory error.

3. **API usage**

```bash
# Submit a PDF (async)
curl -F "file=@document.pdf" http://localhost:8000/jobs

# Poll status
curl http://localhost:8000/jobs/<job_id>

# Retrieve markdown when complete
curl http://localhost:8000/jobs/<job_id>/result.txt

# Optional: inspect jobs in the browser
# Visit http://localhost:8000/admin
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | – | SQLAlchemy connection string for PostgreSQL |
| `OPENROUTER_API_KEY` | – | API key for OpenRouter |
| `OPENROUTER_MODEL` | `google/gemini-2.0-flash-001` | Model used for image-to-markdown conversion |
| `PAGE_IMAGE_WIDTH` | `1500` | Target image width before sending to OpenRouter |
| `PAGE_IMAGE_DPI` | `300` | PDF rasterization DPI |
| `MAX_CONCURRENT_PAGES` | `8` | Page-level concurrency per job |
| `JOB_POLL_INTERVAL_SECONDS` | `2.0` | Delay between jobs once processing completes |
| `JOB_IDLE_SLEEP_SECONDS` | `5.0` | Sleep duration when no queued jobs are found |
| `STORAGE_ROOT` | `/data` | Root directory for storing uploaded PDFs |
| `PDF_SUBDIR` | `pdfs` | Subdirectory (under storage root) for PDFs |

The included `docker-compose.yml` mirrors these processing defaults and enables debug logging (`LOG_LEVEL=DEBUG`).

## Development

```bash
pip install -r requirements.txt
uvicorn pdf_tucano.app:app --reload
```

The worker starts automatically with the FastAPI application.

## Notes

- Poppler must be installed (handled in the Docker image) for `pdf2image`.
- To reset all data, clear the `jobs` and `pages` tables and remove stored PDFs from the volume.
- The service returns HTTP 409 if you request results before a job is complete.
