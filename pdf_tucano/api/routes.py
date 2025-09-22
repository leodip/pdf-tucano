from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional
from uuid import UUID, uuid4

import PyPDF2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from pdf_tucano.api.schemas import (
    JobCreateResponse,
    JobResultResponse,
    JobStatusResponse,
    JobStatusWithPages,
    PageStatusItem,
)
from pdf_tucano.db.models import Job, JobStatus, Page, PageStatus
from pdf_tucano.db.session import SessionLocal
from pdf_tucano.storage.manager import StorageManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_db_session() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    file: UploadFile = File(..., description="PDF file to convert"),
    db: Session = Depends(get_db_session),
) -> JobCreateResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if file.content_type not in {"application/pdf", "application/x-pdf", "application/octet-stream"}:
        logger.warning("Unexpected content type", extra={"content_type": file.content_type})

    try:
        reader = PyPDF2.PdfReader(BytesIO(content))
        total_pages = len(reader.pages)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to read PDF", extra={"filename": file.filename})
        raise HTTPException(status_code=400, detail="Invalid PDF file") from exc

    if total_pages == 0:
        raise HTTPException(status_code=400, detail="PDF contains no pages")

    job_id = uuid4()
    storage = StorageManager()
    pdf_path = storage.save_pdf(str(job_id), content, file.filename)

    job = Job(
        id=job_id,
        original_filename=file.filename,
        pdf_path=str(pdf_path),
        status=JobStatus.QUEUED,
        total_pages=total_pages,
        completed_pages=0,
    )
    db.add(job)

    for page_number in range(1, total_pages + 1):
        page = Page(
            id=uuid4(),
            job_id=job_id,
            page_number=page_number,
            status=PageStatus.PENDING,
        )
        db.add(page)

    db.commit()

    logger.info("Job queued", extra={"job_id": str(job_id), "pages": total_pages})
    return JobCreateResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: UUID,
    include_pages: bool = False,
    db: Session = Depends(get_db_session),
) -> JobStatusResponse | JobStatusWithPages:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    base = JobStatusResponse(
        job_id=job.id,
        status=job.status,
        total_pages=job.total_pages,
        completed_pages=job.completed_pages,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        total_cost=job.total_cost,
        cost_currency=job.cost_currency,
        prompt_tokens=job.prompt_tokens,
        completion_tokens=job.completion_tokens,
        total_tokens=job.total_tokens,
    )

    if not include_pages:
        return base

    stmt = (
        select(
            Page.page_number,
            Page.status,
            Page.error_message,
            Page.generation_id,
            Page.prompt_tokens,
            Page.completion_tokens,
            Page.total_tokens,
            Page.total_cost,
            Page.cost_currency,
            Page.generation_stats_synced_at,
        )
        .where(Page.job_id == job_id)
        .order_by(Page.page_number)
    )
    pages = [
        PageStatusItem(
            page_number=row.page_number,
            status=row.status,
            error_message=row.error_message,
            generation_id=row.generation_id,
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            total_tokens=row.total_tokens,
            total_cost=row.total_cost,
            cost_currency=row.cost_currency,
            generation_stats_synced_at=row.generation_stats_synced_at,
        )
        for row in db.execute(stmt)
    ]
    return JobStatusWithPages(**base.dict(), pages=pages)


@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: UUID, db: Session = Depends(get_db_session)) -> JobResultResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == JobStatus.FAILED:
        return JobResultResponse(
            job_id=job.id,
            status=job.status,
            markdown=None,
            error_message=job.error_message,
            total_cost=job.total_cost,
            cost_currency=job.cost_currency,
            prompt_tokens=job.prompt_tokens,
            completion_tokens=job.completion_tokens,
            total_tokens=job.total_tokens,
        )

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed yet")

    return JobResultResponse(
        job_id=job.id,
        status=job.status,
        markdown=job.result_markdown,
        error_message=None,
        total_cost=job.total_cost,
        cost_currency=job.cost_currency,
        prompt_tokens=job.prompt_tokens,
        completion_tokens=job.completion_tokens,
        total_tokens=job.total_tokens,
    )


@router.get("/jobs/{job_id}/result.txt", response_class=PlainTextResponse)
def download_markdown(job_id: UUID, db: Session = Depends(get_db_session)) -> PlainTextResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed yet")
    return PlainTextResponse(content=job.result_markdown or "", media_type="text/plain")


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
