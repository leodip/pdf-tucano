from __future__ import annotations

from fastapi import FastAPI

from pdf_tucano.api import admin
from pdf_tucano.api.routes import router
from pdf_tucano.config import get_settings
from pdf_tucano.db.init import init_db
from pdf_tucano.logging_config import configure_logging
from pdf_tucano.workers.runner import JobRunner

settings = get_settings()
_, fallback = configure_logging(settings.log_level)
if fallback:
    import logging

    logging.getLogger(__name__).warning(
        "Unknown LOG_LEVEL '%s'; falling back to INFO",
        settings.log_level,
    )

app = FastAPI(title="pdf-tucano", version="1.0.0")
job_runner = JobRunner()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    job_runner.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    job_runner.stop()


app.include_router(router)
app.include_router(admin.router)
