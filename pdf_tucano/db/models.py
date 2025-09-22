from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PageStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True)
    original_filename = Column(String(255), nullable=True)
    pdf_path = Column(String(1024), nullable=False)
    status = Column(Enum(JobStatus, name="job_status_enum"), nullable=False, default=JobStatus.QUEUED)
    total_pages = Column(Integer, nullable=False)
    completed_pages = Column(Integer, nullable=False, default=0)
    result_markdown = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    prompt_tokens = Column(Integer, nullable=False, default=0, server_default="0")
    completion_tokens = Column(Integer, nullable=False, default=0, server_default="0")
    total_tokens = Column(Integer, nullable=False, default=0, server_default="0")
    total_cost = Column(Numeric(18, 6), nullable=False, default=0, server_default="0")
    cost_currency = Column(String(8), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    pages = relationship("Page", back_populates="job", cascade="all, delete-orphan", passive_deletes=True)


class Page(Base):
    __tablename__ = "pages"

    id = Column(UUID(as_uuid=True), primary_key=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    status = Column(Enum(PageStatus, name="page_status_enum"), nullable=False, default=PageStatus.PENDING)
    markdown_content = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    error_message = Column(Text, nullable=True)
    generation_id = Column(String(255), nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    total_cost = Column(Numeric(18, 6), nullable=True)
    cost_currency = Column(String(8), nullable=True)
    generation_stats_attempts = Column(Integer, nullable=False, default=0, server_default="0")
    generation_stats_last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    generation_stats_next_attempt_at = Column(DateTime(timezone=True), nullable=True)
    generation_stats_synced_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("Job", back_populates="pages")

    __mapper_args__ = {
        "eager_defaults": True,
    }
