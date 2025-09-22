import logging

from sqlalchemy import inspect

from pdf_tucano.db.models import Base
from pdf_tucano.db.session import engine


logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create database tables if they do not exist."""
    inspector = inspect(engine)
    if not inspector.has_table("jobs"):
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema created")
    else:
        logger.debug("Database schema already up to date")
