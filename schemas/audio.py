from pydantic import BaseModel
from typing import Optional, List

class AudioAnalysis(BaseModel):
    speech: bool
    transcript: Optional[str] = None
    music: bool = False
    music_mood: Optional[str] = None
    ambient: List[str] = []
    errors: List[str] = []
