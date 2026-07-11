from typing import List, Dict, Any
from schemas.timeline import TimelineEvent
from schemas.vision import Scene
from fusion.merger import merge_and_sort_timeline

def build_timeline(
    audio_segments: List[Dict[str, Any]],
    scenes: List[Scene]
) -> List[TimelineEvent]:
    """
    Orchestrates timeline creation by collecting, merging, sorting,
    and deduplicating audio and visual events.
    """
    return merge_and_sort_timeline(audio_segments, scenes)
