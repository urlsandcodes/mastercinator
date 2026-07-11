from pydantic import BaseModel
from typing import Literal

class TimelineEvent(BaseModel):
    time_seconds: float
    time_display: str            # "00:04"
    event: str
    source: Literal["audio", "vision", "ocr"]
