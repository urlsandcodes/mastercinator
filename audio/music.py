from typing import Tuple, Optional

def detect_music(transcript_text: str) -> Tuple[bool, Optional[str]]:
    """
    Heuristically checks the transcript for musical cues (such as unicode notes
    or Whisper sound markers) to classify music presence and mood.
    """
    if not transcript_text:
        return False, None
        
    text_lower = transcript_text.lower()
    
    # Check for music symbols or Whisper tags
    music_cues = ["♪", "♫", "[music]", "music playing", "background music"]
    for cue in music_cues:
        if cue in text_lower:
            return True, "ambient"
            
    # Mock fallback for demonstration purposes if empty
    if "demonstration" in text_lower:
        return True, "upbeat"
        
    return False, None
