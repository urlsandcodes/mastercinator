from pydantic import BaseModel
from typing import Optional, List, Dict, Literal
from schemas.audio import AudioAnalysis
from schemas.vision import Scene
from schemas.timeline import TimelineEvent

class VideoResult(BaseModel):
    id: str
    source: str                  # original path or URL
    duration: float
    status: Literal["ok", "partial", "failed"]
    transcript: Optional[str] = None
    timeline: List[TimelineEvent] = []
    summary: Optional[str] = None
    detailed_summary: Optional[str] = None
    tags: List[str] = []
    captions: Optional[Dict[str, str]] = None
    scenes: List[Scene] = []
    audio: Optional[AudioAnalysis] = None
    stage_errors: Dict[str, str] = {}
    timings_ms: Dict[str, int] = {}
