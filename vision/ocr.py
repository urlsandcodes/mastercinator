from typing import List
import re
import json
from app.logger import logger

def extract_ocr_from_vlm_text(vlm_text: str) -> List[str]:
    """
    Parses visual text strings detected by the VLM from the frame result.
    Matches lines starting with 'OCR:' and processes the following array.
    """
    if not vlm_text:
        return []
        
    matches = re.findall(r"^\s*[-*•#\d.]*\s*OCR:\s*(.*)$", vlm_text, re.IGNORECASE | re.MULTILINE)
    if not matches:
        return []
        
    raw_str = matches[-1].strip()
    parsed_list = []
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, list):
            parsed_list = [str(x) for x in parsed]
        elif isinstance(parsed, str):
            parsed_list = [x.strip() for x in parsed.split(",") if x.strip()]
    except Exception:
        # Fallback to character scrubbing and comma parsing if JSON format was irregular
        cleaned = re.sub(r'[\[\]"\'`]', '', raw_str)
        if cleaned:
            parsed_list = [x.strip() for x in cleaned.split(",") if x.strip()]
            
    # Filter out VLM meta-talk/explanations
    filtered = []
    meta_indicators = [
        "dont see", "don't see", "no text", "no visible", "no readable",
        "need to look", "explain", "i do not see", "there is no", "theres no",
        "text on screen", "clear text in the image", "unable to read",
        "or readable documents"
    ]
    for text in parsed_list:
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in meta_indicators):
            continue
        # If it's a long sentence explaining lack of text, drop it
        if len(text.split()) > 6 and ("text" in text_lower or "image" in text_lower or "visible" in text_lower):
            continue
        filtered.append(text)
        
    return filtered
