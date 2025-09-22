from __future__ import annotations

from pathlib import Path
from typing import Generator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pdf_tucano.db.models import Job, Page
from pdf_tucano.db.session import SessionLocal


def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/", response_class=HTMLResponse)
def list_jobs(request: Request, db: Session = Depends(get_db_session)) -> HTMLResponse:
    total_count = db.execute(select(func.count()).select_from(Job)).scalar_one()
    jobs = (
        db.execute(
            select(Job).order_by(Job.created_at.desc()).limit(50)
        ).scalars().all()
    )
    context = {
        "request": request,
        "jobs": jobs,
        "total_count": total_count,
    }
    return templates.TemplateResponse("admin/jobs.html", context)


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: UUID, db: Session = Depends(get_db_session)) -> HTMLResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pages = (
        db.execute(
            select(Page).where(Page.job_id == job_id).order_by(Page.page_number)
        ).scalars().all()
    )

    context = {
        "request": request,
        "job": job,
        "pages": pages,
    }
    return templates.TemplateResponse("admin/job_detail.html", context)
