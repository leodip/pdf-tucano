from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from pdf_tucano.config import get_settings
from pdf_tucano.core.pdf_processor import PdfProcessor
from pdf_tucano.db.models import Job, JobStatus
from pdf_tucano.db.session import session_scope

logger = logging.getLogger(__name__)


class JobRunner:
    """Background worker that polls for queued jobs and processes them."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.processor = PdfProcessor(session_scope)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="pdf-tucano-runner", daemon=True)
        self._thread.start()
        logger.info("Job runner started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Job runner stopped")

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        poll_interval = self.settings.job_poll_interval_seconds
        idle_sleep = self.settings.job_idle_sleep_seconds

        while not self._stop_event.is_set():
            refresh_timer = time.perf_counter()
            try:
                refreshed = self.processor.refresh_pending_generation_stats()
                refresh_duration = round(time.perf_counter() - refresh_timer, 3)
                if refreshed:
                    logger.info(
                        "Refreshed generation stats",
                        extra={"pages": refreshed, "duration_seconds": refresh_duration},
                    )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception(
                    "Generation stats refresh failed",
                    extra={
                        "error": str(exc),
                        "duration_seconds": round(time.perf_counter() - refresh_timer, 3),
                    },
                )

            job_id = self._fetch_next_job_id()
            if job_id is None:
                logger.debug(
                    "No queued jobs found",
                    extra={"sleep_seconds": idle_sleep},
                )
                time.sleep(idle_sleep)
                continue

            logger.info("Processing queued job", extra={"job_id": str(job_id)})
            job_timer = time.perf_counter()
            success, error = self.processor.process_job(job_id)
            job_duration = round(time.perf_counter() - job_timer, 3)
            if not success:
                logger.error(
                    "Job processing failed",
                    extra={"job_id": str(job_id), "error": error, "duration_seconds": job_duration},
                )
            else:
                logger.info(
                    "Job processing finished",
                    extra={"job_id": str(job_id), "duration_seconds": job_duration},
                )
            time.sleep(poll_interval)

    def _fetch_next_job_id(self) -> Optional[UUID]:
        with session_scope() as session:
            stmt = (
                select(Job)
                .where(Job.status == JobStatus.QUEUED)
                .order_by(Job.created_at)
                .with_for_update(skip_locked=True)
            )
            job = session.execute(stmt).scalars().first()
            if not job:
                return None

            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            session.add(job)
            session.commit()
            logger.info(
                "Promoted job to processing",
                extra={
                    "job_id": str(job.id),
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                },
            )
            return job.id
