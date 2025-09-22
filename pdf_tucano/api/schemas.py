from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from pdf_tucano.db.models import JobStatus, PageStatus


class JobCreateResponse(BaseModel):
    job_id: UUID
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    total_pages: int
    completed_pages: int
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_cost: Optional[Decimal]
    cost_currency: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class PageStatusItem(BaseModel):
    page_number: int
    status: PageStatus
    error_message: Optional[str]
    generation_id: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    total_cost: Optional[Decimal]
    cost_currency: Optional[str]
    generation_stats_synced_at: Optional[datetime]


class JobStatusWithPages(JobStatusResponse):
    pages: list[PageStatusItem]


class JobResultResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    markdown: Optional[str]
    error_message: Optional[str]
    total_cost: Optional[Decimal]
    cost_currency: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
