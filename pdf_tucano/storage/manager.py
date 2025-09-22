from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from pdf_tucano.config import get_settings


class StorageManager:
    """Filesystem storage helper for PDFs and derived assets."""

    def __init__(self) -> None:
        settings = get_settings()
        self.root = settings.storage_root
        self.pdf_dir = self.root / settings.pdf_subdir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def pdf_path(self, job_id: str, original_filename: str | None = None) -> Path:
        suffix = Path(original_filename).suffix if original_filename else ".pdf"
        return self.pdf_dir / f"{job_id}{suffix}"

    def save_pdf(self, job_id: str, data: bytes, original_filename: str | None = None) -> Path:
        path = self.pdf_path(job_id, original_filename)
        with open(path, "wb") as handle:
            handle.write(data)
        return path
