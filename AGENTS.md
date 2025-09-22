# pdf-tucano Agents

This document explains the autonomous components ("agents") that collaborate to turn uploaded PDFs into Markdown. Each agent encapsulates a focused responsibility and communicates through PostgreSQL, the filesystem, and HTTP APIs.

## System Overview
- Clients create jobs through FastAPI endpoints. Jobs and per-page tasks are registered in PostgreSQL and PDFs are persisted on disk.
- A background worker thread polls queued jobs, promotes them to `processing`, and hands off work to the PDF processor.
- The processor rasterizes each page, calls an LLM via OpenRouter to transcribe the page image, and aggregates Markdown back into the job record.
- Status, results, and lightweight admin views expose job progress for users and operators.

## Agents

### HTTP Gateway Agent (`pdf_tucano.api.routes`)
- Accepts file uploads at `/jobs`, validates PDFs with `PyPDF2`, and records job + page rows with initial `queued`/`pending` statuses.
- Redirects storage to the `StorageManager`, ensuring the PDF is written under `${STORAGE_ROOT}/${PDF_SUBDIR}` with the job UUID as filename.
- Exposes `GET /jobs/{id}` and `GET /jobs/{id}/result(.txt)` for polling status or retrieving Markdown, returning `409` when results are requested before completion and bubbling job-level errors.
- Delegates DB access through `SessionLocal`, closing sessions per request to avoid leaks.

### Storage Agent (`pdf_tucano.storage.manager.StorageManager`)
- Creates the storage hierarchy on construction using `STORAGE_ROOT` (default `/data`).
- Derives deterministic filenames from the job UUID, preserving the source suffix when available.
- Persists PDFs atomically via `save_pdf`, enabling the processor to reload raw bytes later.

### Persistence Agent (`pdf_tucano.db`)
- SQLAlchemy models `Job` and `Page` capture job metadata, per-page status, Markdown fragments, timestamps, and error details.
- `SessionLocal`/`session_scope` provide scoped transactions with automatic commit/rollback, ensuring state transitions remain consistent across agents.
- Status enums (`JobStatus`, `PageStatus`) codify the lifecycle: `queued → processing → completed` (or `failed`).

### Job Runner Agent (`pdf_tucano.workers.runner.JobRunner`)
- Boots on FastAPI startup and runs as a daemon thread until shutdown.
- Polls the DB for the oldest `queued` job (`SELECT ... FOR UPDATE SKIP LOCKED`), promotes it to `processing`, and timestamps `started_at`.
- Sleeps according to configuration (`JOB_POLL_INTERVAL_SECONDS`, `JOB_IDLE_SLEEP_SECONDS`) when idle or between runs.
- Delegates heavy lifting to the `PdfProcessor` and logs successes/failures for observability.

### PDF Processing Agent (`pdf_tucano.core.pdf_processor.PdfProcessor`)
- Loads the stored PDF bytes, enumerates pending/failed pages, and spins a `ThreadPoolExecutor` limited by `MAX_CONCURRENT_PAGES`.
- For each page: rasterizes via `pdf2image.convert_from_bytes` at `PAGE_IMAGE_DPI`, resizes to `PAGE_IMAGE_WIDTH`, converts to PNG, and base64-encodes the image payload.
- Invokes the OpenRouter client per page, updates page rows to `processing`/`completed`, increments `Job.completed_pages`, and captures Markdown fragments.
- Aggregates ordered page Markdown as `## {page_label} {N}` sections (configurable via `PAGE_HEADING_LABEL`, defaulting to `Page`), storing the combined text + job completion timestamps. On any page failure, marks the job `failed` with the first error message.

### Vision-to-Markdown Agent (`pdf_tucano.core.openrouter_client.OpenRouterClient`)
- Wraps the OpenRouter `/chat/completions` API using `requests` with retry/backoff (`backoff.expo`, 5 tries).
- Requires `OPENROUTER_API_KEY` and chooses the model via `OPENROUTER_MODEL` (default `google/gemini-2.0-flash-001`).
- Sends a structured prompt (Portuguese instructions, no markdown fences) alongside the PNG data URL, returning the model's raw Markdown output.

### Admin UI Agent (`pdf_tucano.api.admin` + `templates/`)
- Provides `/admin` views for operators: paginated job lists, job detail pages with per-page status and Markdown excerpts.
- Shares database dependencies with the HTTP gateway but renders via Jinja2 templates for quick inspection.

## Workflow
1. **Submission** – `POST /jobs` validates the PDF, saves it via Storage Agent, inserts job/pages (`queued`/`pending`).
2. **Scheduling** – Job Runner Agent locks the next queued job, flips status to `processing`, and triggers the processor.
3. **Per-page processing** – PDF Processing Agent rasterizes pages, hands base64 PNGs to the OpenRouter agent, and updates page rows.
4. **Aggregation** – When all pages succeed, Markdown is concatenated, job status becomes `completed`, timestamps + counts are filled.
5. **Delivery** – Clients poll `/jobs/{id}` for progress or `/jobs/{id}/result(.txt)` for the final Markdown. Admin UI mirrors this data.

## Error Handling & Retries
- Page-level exceptions mark the page `failed` and set `Job.error_message`; the job transitions to `failed` once processing finishes.
- Runner logs failures and moves on; the job remains `failed` until manually retriggered (e.g., by resetting statuses in the DB).
- OpenRouter transient errors trigger exponential backoff before surfacing as page failures.
- Missing PDFs or empty uploads are caught early by the gateway agent, returning `400 Bad Request`.

## Configuration Surface
Key environment variables (see `pdf_tucano/config.py`):
- `DATABASE_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- `PAGE_IMAGE_WIDTH`, `PAGE_IMAGE_DPI`, `MAX_CONCURRENT_PAGES`
- `JOB_POLL_INTERVAL_SECONDS`, `JOB_IDLE_SLEEP_SECONDS`
- `STORAGE_ROOT`, `PDF_SUBDIR`, `CLEANUP_COMPLETED_AFTER_HOURS`
- `PAGE_HEADING_LABEL`

Changes require service restart because `Settings` is cached (`lru_cache`).

## Extending Agents
- **Alternate transcription models** – Implement a new client following `OpenRouterClient`'s interface and inject via dependency control.
- **Task retries** – Add logic in `PdfProcessor` to requeue failed pages or allow manual retrigger endpoints.
- **Post-processing** – Extend aggregation to clean Markdown, split into chapters, or store additional metadata per page.
- **Cleanup agent** – Introduce a scheduled task that honors `CLEANUP_COMPLETED_AFTER_HOURS` to purge old PDFs/records.

These agents operate together to deliver reliable PDF→Markdown conversion while keeping responsibilities isolated and composable.
