from pydantic import BaseModel
from typing import Optional, List

class Scene(BaseModel):
    scene_id: int
    start: float
    end: float
    description: Optional[str] = None      # from Gemma 4 (or configured VLM)
    objects: List[str] = []
    actions: List[str] = []
    ocr: List[str] = []
    errors: List[str] = []
