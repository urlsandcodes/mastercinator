import logging
import sys
from contextvars import ContextVar
from typing import Optional

# ContextVar to store the correlation ID (video_id) for the current task
video_id_var: ContextVar[Optional[str]] = ContextVar("video_id", default=None)

class CorrelationFilter(logging.Filter):
    """Injects the video correlation ID into the log record."""
    def filter(self, record):
        vid = video_id_var.get()
        record.video_id = vid if vid else "system"
        return True

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("video_intelligence")
    logger.setLevel(logging.INFO)
    
    # Prevent handler duplication if the logger is reconfigured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Format: timestamp - [level] - [video_id] - message
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(video_id)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.addFilter(CorrelationFilter())
    return logger

logger = setup_logger()
