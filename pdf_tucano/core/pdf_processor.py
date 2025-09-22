from __future__ import annotations

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import UUID

import PyPDF2
from pdf2image import convert_from_bytes
from PIL import Image
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from pdf_tucano.config import get_settings
from pdf_tucano.core.openrouter_client import GenerationStats, OpenRouterClient, OpenRouterCompletion
from pdf_tucano.db.models import Job, JobStatus, Page, PageStatus

logger = logging.getLogger(__name__)


def _elapsed_seconds(start: float) -> float:
    """Return rounded wall-clock seconds since ``start``."""

    return round(time.perf_counter() - start, 3)


class PdfProcessor:
    """Convert PDFs to markdown using OpenRouter, mirroring legacy workflow."""

    MAX_GENERATION_FETCH_ATTEMPTS = 8
    MAX_GENERATION_REFRESH_BATCH = 8
    GENERATION_REFRESH_DELAYS = (30, 60, 120, 240, 480)

    def __init__(self, session_factory: Callable[[], AbstractContextManager[Session]]) -> None:
        self.session_factory = session_factory
        self.settings = get_settings()
        self.api_client = OpenRouterClient()

    @staticmethod
    def _sanitize_markdown(text: Optional[str]) -> str:
        """Remove disallowed markdown fences before persistence."""

        if not text:
            return ""

        sanitized = text.replace("```markdown", "")
        sanitized = sanitized.replace("```", "")
        return sanitized

    def process_job(self, job_id: UUID) -> Tuple[bool, Optional[str]]:
        """Process a job and return success flag with optional error message."""
        job_timer = time.perf_counter()
        job = self._get_job(job_id)
        if job is None:
            logger.warning("Job missing in database", extra={"job_id": str(job_id)})
            return False, "Job not found"

        pdf_path = Path(job.pdf_path)
        if not pdf_path.exists():
            logger.error(
                "PDF file missing",
                extra={
                    "job_id": str(job_id),
                    "path": str(pdf_path),
                    "duration_seconds": _elapsed_seconds(job_timer),
                },
            )
            return False, "PDF file not found"

        pdf_bytes = pdf_path.read_bytes()
        total_pages = job.total_pages

        start_time = datetime.now(timezone.utc)
        queue_age_seconds: Optional[float] = None
        if job.created_at:
            queue_age_seconds = round((start_time - job.created_at).total_seconds(), 3)

        processing_context = {
            "job_id": str(job_id),
            "total_pages": total_pages,
            "status": job.status.value,
            "completed_pages": job.completed_pages,
        }
        if queue_age_seconds is not None:
            processing_context["queue_age_seconds"] = queue_age_seconds

        logger.info(
            "Processing job",
            extra=processing_context,
        )

        success, markdown, error_message = self._process_pdf(job_id, pdf_bytes, total_pages)
        duration_seconds = _elapsed_seconds(job_timer)
        if not success:
            logger.error(
                "Job processing failed",
                extra={
                    "job_id": str(job_id),
                    "error": error_message,
                    "duration_seconds": duration_seconds,
                },
            )
            return False, error_message

        sanitized_markdown = self._sanitize_markdown(markdown)

        # Persist combined markdown on success
        with self.session_factory() as session:
            db_job = session.get(Job, job_id)
            if db_job:
                db_job.result_markdown = sanitized_markdown
                db_job.completed_pages = db_job.total_pages
                db_job.status = JobStatus.COMPLETED
                db_job.completed_at = datetime.now(timezone.utc)
                db_job.error_message = None
                session.add(db_job)

        logger.info(
            "Job completed",
            extra={
                "job_id": str(job_id),
                "total_pages": total_pages,
                "result_length": len(sanitized_markdown),
                "duration_seconds": duration_seconds,
            },
        )
        return True, None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _get_job(self, job_id: UUID) -> Optional[Job]:
        with self.session_factory() as session:
            return session.get(Job, job_id)

    def _get_pages(self, session: Session, job_id: UUID) -> List[Tuple[UUID, int]]:
        stmt = (
            select(Page.id, Page.page_number)
            .where(Page.job_id == job_id, Page.status.in_([PageStatus.PENDING, PageStatus.FAILED]))
            .order_by(Page.page_number)
        )
        return [(row.id, row.page_number) for row in session.execute(stmt)]

    def _process_pdf(
        self,
        job_id: UUID,
        pdf_bytes: bytes,
        total_pages: int,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        pdf_timer = time.perf_counter()
        with self.session_factory() as session:
            pages = self._get_pages(session, job_id)

        if not pages:
            logger.error(
                "Job has no pages to process",
                extra={"job_id": str(job_id), "duration_seconds": _elapsed_seconds(pdf_timer)},
            )
            return False, None, "No pages to process"

        results: List[Tuple[int, Optional[str], Optional[str]]] = []

        logger.debug(
            "Dispatching pages",
            extra={
                "job_id": str(job_id),
                "page_count": len(pages),
                "max_workers": self.settings.max_concurrent_pages,
            },
        )

        with ThreadPoolExecutor(max_workers=self.settings.max_concurrent_pages) as executor:
            future_to_page = {
                executor.submit(
                    self._process_single_page,
                    pdf_bytes,
                    job_id,
                    page_id,
                    page_number,
                ): (page_id, page_number)
                for page_id, page_number in pages
            }

            for future in as_completed(future_to_page):
                page_id, page_number = future_to_page[future]
                try:
                    markdown, error = future.result()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Unhandled exception in page processing",
                        extra={"job_id": str(job_id), "page_number": page_number},
                    )
                    markdown, error = None, str(exc)

                results.append((page_number, markdown, error))

        failed = [item for item in results if item[1] is None]
        if failed:
            error_msg = failed[0][2] or "Page processing failed"
            logger.error(
                "Job failed",
                extra={
                    "job_id": str(job_id),
                    "failed_pages": [page for page, _, _ in failed],
                    "error": error_msg,
                    "duration_seconds": _elapsed_seconds(pdf_timer),
                },
            )
            with self.session_factory() as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = error_msg
                    job.completed_at = datetime.now(timezone.utc)
                    session.add(job)
            return False, None, error_msg

        # Sort by page order and compose markdown content
        results.sort(key=lambda item: item[0])
        heading_label = (self.settings.page_heading_label or "").strip()
        heading_prefix = f"## {heading_label} " if heading_label else "## "
        combined = "\n".join(
            f"\n\n{heading_prefix}{page_number}\n\n{markdown}"
            for page_number, markdown, _ in results
        )
        combined = self._sanitize_markdown(combined)
        logger.debug(
            "Aggregated markdown",
            extra={
                "job_id": str(job_id),
                "page_count": len(results),
                "result_length": len(combined),
                "duration_seconds": _elapsed_seconds(pdf_timer),
            },
        )
        return True, combined, None

    def _process_single_page(
        self,
        pdf_bytes: bytes,
        job_id: UUID,
        page_id: UUID,
        page_number: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        page_timer = time.perf_counter()
        logger.info(
            "Processing page",
            extra={
                "job_id": str(job_id),
                "page_number": page_number,
                "page_id": str(page_id),
            },
        )
        self._update_page_status(page_id, PageStatus.PROCESSING, None)

        completion: Optional[OpenRouterCompletion] = None
        render_seconds: Optional[float] = None
        llm_seconds: Optional[float] = None
        stats_seconds: Optional[float] = None
        try:
            render_timer = time.perf_counter()
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.settings.page_image_dpi,
                first_page=page_number,
                last_page=page_number,
            )
            render_seconds = _elapsed_seconds(render_timer)
            if not images:
                raise RuntimeError(f"No image generated for page {page_number}")

            image = images[0]
            resized = self._resize_image(image)

            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            llm_timer = time.perf_counter()
            completion = self.api_client.convert_image_to_markdown(encoded_image)
            llm_seconds = _elapsed_seconds(llm_timer)
            if not completion.markdown:
                raise RuntimeError("OpenRouter returned empty response")

            logger.debug(
                "Received OpenRouter completion",
                extra={
                    "job_id": str(job_id),
                    "page_number": page_number,
                    "generation_id": completion.generation_id,
                    "markdown_length": len(completion.markdown),
                    "llm_seconds": llm_seconds,
                },
            )

            stats: Optional[GenerationStats] = None
            if completion.generation_id:
                try:
                    stats_timer = time.perf_counter()
                    stats = self.api_client.fetch_generation_stats(completion.generation_id)
                    stats_seconds = _elapsed_seconds(stats_timer)
                except Exception as stats_exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Failed to fetch generation stats: %s",
                        stats_exc,
                        extra={
                            "job_id": str(job_id),
                            "page_number": page_number,
                            "generation_id": completion.generation_id,
                        },
                    )
                else:
                    logger.debug(
                        "Applied generation stats",
                        extra={
                            "job_id": str(job_id),
                            "page_number": page_number,
                            "generation_id": completion.generation_id,
                            "total_cost": str(stats.total_cost) if stats and stats.total_cost is not None else None,
                            "currency": stats.currency if stats else None,
                            "stats_seconds": stats_seconds,
                        },
                    )

            if stats is None or stats.total_cost is None:
                logger.info(
                    "Generation stats missing cost; using zero",
                    extra={
                        "job_id": str(job_id),
                        "page_number": page_number,
                        "generation_id": completion.generation_id,
                        "has_stats": stats is not None,
                    },
                )

            sanitized_markdown = self._sanitize_markdown(completion.markdown)

            self._complete_page(page_id, job_id, sanitized_markdown, completion.generation_id, stats)
            logger.info(
                "Page converted",
                extra={
                    "job_id": str(job_id),
                    "page_number": page_number,
                    "page_id": str(page_id),
                    "generation_id": completion.generation_id,
                    "markdown_length": len(sanitized_markdown),
                    "duration_seconds": _elapsed_seconds(page_timer),
                    "render_seconds": render_seconds,
                    "llm_seconds": llm_seconds,
                    "stats_seconds": stats_seconds,
                },
            )
            return sanitized_markdown, None

        except Exception as exc:  # pragma: no cover - defensive logging
            error_text = str(exc)
            logger.exception(
                "Page processing error",
                extra={
                    "job_id": str(job_id),
                    "page_number": page_number,
                    "error": error_text,
                    "generation_id": completion.generation_id if completion else None,
                    "duration_seconds": _elapsed_seconds(page_timer),
                    "render_seconds": render_seconds,
                    "llm_seconds": llm_seconds,
                },
            )
            self._fail_page(page_id, job_id, error_text)
            if "float object has no attribute 'get'" in error_text:
                logger.warning(
                    "OpenRouter stats payload shape unexpected",
                    extra={
                        "job_id": str(job_id),
                        "generation_id": completion.generation_id if completion else None,
                        "page_number": page_number,
                    },
                )
            return None, error_text

    def _resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= self.settings.page_image_width:
            return image
        new_width = self.settings.page_image_width
        new_height = int(height * (new_width / width))
        return image.resize((new_width, new_height), Image.LANCZOS)

    def _update_page_status(
        self,
        page_id: UUID,
        status: PageStatus,
        error: Optional[str],
    ) -> None:
        with self.session_factory() as session:
            page = session.get(Page, page_id)
            if page:
                page.status = status
                page.error_message = error
                session.add(page)
                logger.debug(
                    "Page status updated",
                    extra={
                        "page_id": str(page_id),
                        "job_id": str(page.job_id),
                        "status": status.value,
                        "error": error,
                    },
                )

    def _complete_page(
        self,
        page_id: UUID,
        job_id: UUID,
        markdown: str,
        generation_id: Optional[str],
        stats: Optional[GenerationStats],
    ) -> None:
        sanitized_markdown = self._sanitize_markdown(markdown)
        with self.session_factory() as session:
            page = session.get(Page, page_id)
            if page:
                page.status = PageStatus.COMPLETED
                page.markdown_content = sanitized_markdown
                page.error_message = None
                page.generation_id = generation_id
                page.generation_stats_attempts = page.generation_stats_attempts or 0
                if stats:
                    page.generation_stats_synced_at = datetime.now(timezone.utc)
                    page.prompt_tokens = stats.prompt_tokens
                    page.completion_tokens = stats.completion_tokens
                    page.total_tokens = stats.total_tokens
                    page.total_cost = stats.total_cost
                    page.cost_currency = stats.currency
                session.add(page)
                logger.debug(
                    "Page completion persisted",
                    extra={
                        "page_id": str(page_id),
                        "job_id": str(job_id),
                        "generation_id": generation_id,
                        "has_stats": stats is not None,
                    },
                )

            updates: Dict[str, Any] = {"completed_pages": Job.completed_pages + 1}

            if stats:
                if stats.prompt_tokens is not None:
                    updates["prompt_tokens"] = func.coalesce(Job.prompt_tokens, 0) + stats.prompt_tokens
                if stats.completion_tokens is not None:
                    updates["completion_tokens"] = func.coalesce(Job.completion_tokens, 0) + stats.completion_tokens
                if stats.total_tokens is not None:
                    updates["total_tokens"] = func.coalesce(Job.total_tokens, 0) + stats.total_tokens
                elif stats.prompt_tokens is not None or stats.completion_tokens is not None:
                    token_sum = (stats.prompt_tokens or 0) + (stats.completion_tokens or 0)
                    updates["total_tokens"] = func.coalesce(Job.total_tokens, 0) + token_sum
                if stats.total_cost is not None:
                    updates["total_cost"] = func.coalesce(Job.total_cost, 0) + stats.total_cost
                if stats.currency:
                    updates["cost_currency"] = stats.currency

            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(**updates)
            )
            logger.debug(
                "Updated job aggregates",
                extra={
                    "job_id": str(job_id),
                    "updates": {k: str(v) for k, v in updates.items()},
                },
            )

    def _fail_page(self, page_id: UUID, job_id: UUID, error: str) -> None:
        with self.session_factory() as session:
            page = session.get(Page, page_id)
            if page:
                page.status = PageStatus.FAILED
                page.error_message = error
                session.add(page)
                logger.warning(
                    "Page marked failed",
                    extra={
                        "page_id": str(page_id),
                        "job_id": str(job_id),
                        "error": error,
                    },
                )

            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(error_message=error)
            )
            logger.warning(
                "Job error recorded",
                extra={"job_id": str(job_id), "error": error},
            )

    # ------------------------------------------------------------------
    # Deferred generation stats retrieval
    # ------------------------------------------------------------------
    def refresh_pending_generation_stats(self) -> int:
        """Attempt to populate generation stats for completed pages missing cost."""
        candidates = self._pending_generation_pages()
        if not candidates:
            logger.debug("No pending generation stats to refresh")
            return 0

        processed = 0
        for page_id, generation_id, job_id, attempts in candidates:
            try:
                stats = self.api_client.fetch_generation_stats(generation_id)
            except Exception as exc:  # pragma: no cover - logging for observability
                self._record_generation_attempt(page_id, success=False)
                logger.warning(
                    "Deferred generation stats fetch failed",
                    extra={
                        "job_id": str(job_id),
                        "page_id": str(page_id),
                        "generation_id": generation_id,
                        "attempts": attempts,
                        "max_attempts": self.MAX_GENERATION_FETCH_ATTEMPTS,
                        "attempts_left": max(self.MAX_GENERATION_FETCH_ATTEMPTS - attempts, 0),
                        "error": str(exc),
                    },
                )
                continue

            self._apply_generation_stats(page_id, job_id, stats)
            self._record_generation_attempt(page_id, success=True)
            processed += 1

        return processed

    def _pending_generation_pages(self) -> List[Tuple[UUID, str, UUID, int]]:
        now = datetime.now(timezone.utc)
        with self.session_factory() as session:
            stmt = (
                select(
                    Page.id,
                    Page.generation_id,
                    Page.job_id,
                    Page.generation_stats_attempts,
                    Page.generation_stats_next_attempt_at,
                )
                .where(
                    Page.status == PageStatus.COMPLETED,
                    Page.generation_id.isnot(None),
                    Page.total_cost.is_(None),
                    Page.generation_stats_attempts < self.MAX_GENERATION_FETCH_ATTEMPTS,
                    (
                        Page.generation_stats_next_attempt_at.is_(None)
                        | (Page.generation_stats_next_attempt_at <= now)
                    ),
                )
                .order_by(Page.generation_stats_next_attempt_at.nullsfirst(), Page.updated_at)
                .limit(self.MAX_GENERATION_REFRESH_BATCH)
            )
            return [
                (
                    row.id,
                    row.generation_id,
                    row.job_id,
                    row.generation_stats_attempts or 0,
                )
                for row in session.execute(stmt)
            ]

    def _record_generation_attempt(self, page_id: UUID, success: bool) -> None:
        with self.session_factory() as session:
            page = session.get(Page, page_id)
            if not page:
                return

            now = datetime.now(timezone.utc)
            attempts = (page.generation_stats_attempts or 0) + 1
            page.generation_stats_attempts = attempts
            page.generation_stats_last_attempt_at = now

            delay_seconds: Optional[int] = None
            if page.total_cost is None:
                delay_index = min(attempts - 1, len(self.GENERATION_REFRESH_DELAYS) - 1)
                delay_seconds = self.GENERATION_REFRESH_DELAYS[delay_index]
                page.generation_stats_next_attempt_at = now + timedelta(seconds=delay_seconds)
            else:
                page.generation_stats_next_attempt_at = None
            session.add(page)

            if attempts >= self.MAX_GENERATION_FETCH_ATTEMPTS and page.total_cost is None:
                logger.error(
                    "Exceeded generation stats attempts",
                    extra={"page_id": str(page_id), "generation_id": page.generation_id},
                )
            else:
                attempts_left = max(self.MAX_GENERATION_FETCH_ATTEMPTS - attempts, 0)
                logger.info(
                    "Recorded generation stats attempt",
                    extra={
                        "page_id": str(page_id),
                        "generation_id": page.generation_id,
                        "attempt": attempts,
                        "max_attempts": self.MAX_GENERATION_FETCH_ATTEMPTS,
                        "attempts_left": attempts_left,
                        "success": success,
                        "next_attempt_in_seconds": delay_seconds,
                        "next_attempt_at": page.generation_stats_next_attempt_at.isoformat()
                        if page.generation_stats_next_attempt_at
                        else None,
                    },
                )

    def _apply_generation_stats(self, page_id: UUID, job_id: UUID, stats: GenerationStats) -> None:
        with self.session_factory() as session:
            page = session.get(Page, page_id)
            if not page:
                return

            now = datetime.now(timezone.utc)
            existing_prompt = page.prompt_tokens or 0
            existing_completion = page.completion_tokens or 0
            existing_total_tokens = page.total_tokens or 0
            existing_cost = page.total_cost or Decimal("0")

            delta_prompt = 0
            delta_completion = 0
            delta_total_tokens = 0
            delta_cost = Decimal("0")

            if stats.prompt_tokens is not None:
                delta_prompt = stats.prompt_tokens - existing_prompt
                page.prompt_tokens = stats.prompt_tokens
            if stats.completion_tokens is not None:
                delta_completion = stats.completion_tokens - existing_completion
                page.completion_tokens = stats.completion_tokens

            if stats.total_tokens is not None:
                delta_total_tokens = stats.total_tokens - existing_total_tokens
                page.total_tokens = stats.total_tokens
            else:
                desired_total = (stats.prompt_tokens or 0) + (stats.completion_tokens or 0)
                delta_total_tokens = desired_total - existing_total_tokens
                if desired_total:
                    page.total_tokens = desired_total

            if stats.total_cost is not None:
                delta_cost = stats.total_cost - existing_cost
                page.total_cost = stats.total_cost
                page.cost_currency = stats.currency or page.cost_currency

            page.generation_stats_synced_at = now
            session.add(page)

            logger.debug(
                "Computed generation deltas",
                extra={
                    "page_id": str(page_id),
                    "generation_id": stats.generation_id,
                    "delta_prompt": delta_prompt,
                    "delta_completion": delta_completion,
                    "delta_total_tokens": delta_total_tokens,
                    "delta_cost": str(delta_cost),
                },
            )

            job = session.get(Job, job_id)
            if job:
                updates: Dict[str, Any] = {}

                if delta_prompt > 0:
                    updates["prompt_tokens"] = func.coalesce(Job.prompt_tokens, 0) + delta_prompt
                if delta_completion > 0:
                    updates["completion_tokens"] = func.coalesce(Job.completion_tokens, 0) + delta_completion
                if delta_total_tokens > 0:
                    updates["total_tokens"] = func.coalesce(Job.total_tokens, 0) + delta_total_tokens
                if delta_cost > 0:
                    updates["total_cost"] = func.coalesce(Job.total_cost, 0) + delta_cost
                    if stats.currency:
                        updates["cost_currency"] = stats.currency

                if updates:
                    session.execute(
                        update(Job)
                        .where(Job.id == job_id)
                        .values(**updates)
                    )
                    logger.debug(
                        "Applied job deltas from generation stats",
                        extra={
                            "job_id": str(job_id),
                            "page_id": str(page_id),
                            "updates": {k: str(v) for k, v in updates.items()},
                        },
                    )

        if stats.total_cost is None:
            logger.info(
                "Generation stats refreshed without cost",
                extra={"page_id": str(page_id), "generation_id": stats.generation_id},
            )
        else:
            logger.debug(
                "Generation stats applied",
                extra={
                    "page_id": str(page_id),
                    "generation_id": stats.generation_id,
                    "total_cost": str(stats.total_cost),
                },
            )
