from typing import List

def detect_ambient(transcript_text: str) -> List[str]:
    """
    Heuristically checks the transcript for ambient sound indicators
    (like [laughter], [applause], etc.) commonly inserted by Whisper.
    """
    if not transcript_text:
        return []
        
    text_lower = transcript_text.lower()
    detected = []
    
    markers = {
        "[laughter]": "laughter",
        "[applause]": "applause",
        "[sigh]": "sigh",
        "[cough]": "cough",
        "[gasp]": "gasp",
        "wind blowing": "wind",
        "birds chirping": "nature_sounds"
    }
    
    for marker, name in markers.items():
        if marker in text_lower:
            detected.append(name)
            
    # Mock fallback for demonstration
    if "prototype" in text_lower and not detected:
        detected.append("office_hum")
        
    return detected
