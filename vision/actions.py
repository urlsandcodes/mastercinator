from typing import List
import re
import json
from app.logger import logger

def extract_actions_from_vlm_text(vlm_text: str) -> List[str]:
    """
    Parses actions identified by the VLM from the frame analysis block.
    Matches lines starting with 'ACTIONS:' (with optional list bullets) and decodes the array.
    """
    if not vlm_text:
        return []
        
    matches = re.findall(r"^\s*[-*•#\d.]*\s*ACTIONS:\s*(.*)$", vlm_text, re.IGNORECASE | re.MULTILINE)
    if not matches:
        return []
        
    raw_str = matches[-1].strip()
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        elif isinstance(parsed, str):
            return [x.strip() for x in parsed.split(",") if x.strip()]
    except Exception:
        # Fallback regex/comma parsing
        cleaned = re.sub(r'[\[\]"\'`]', '', raw_str)
        if cleaned:
            return [x.strip() for x in cleaned.split(",") if x.strip()]
            
    return []

def extract_section_content(text: str, current_section: str, next_sections: List[str]) -> str:
    """
    Helper to extract content of a section up to the next uppercase section header.
    """
    if not text:
        return ""
    # Look for current section header and match everything up to the next header or end of text.
    pattern = rf"{current_section}:\s*(.*?)(?=\b(?:{'|'.join(next_sections)}):\s*|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_description_from_vlm_text(vlm_text: str) -> str:
    """
    Parses the scene description field from the VLM response.
    Gets the last occurrence to tolerate thinking/reasoning blocks.
    """
    if not vlm_text:
        return ""
    desc = extract_section_content(vlm_text, "DESCRIPTION", ["MOOD", "DYNAMIC", "CONTEXT", "ACTIONS", "OBJECTS"])
    if desc:
        return desc
    matches = re.findall(r"^\s*[-*•#\d.]*\s*DESCRIPTION:\s*(.*)$", vlm_text, re.IGNORECASE | re.MULTILINE)
    if matches:
        return matches[-1].strip()
    return vlm_text.strip()

def extract_mood_from_vlm_text(vlm_text: str) -> str:
    """
    Parses the scene mood/emotional tone from the VLM response.
    """
    return extract_section_content(vlm_text, "MOOD", ["DESCRIPTION", "DYNAMIC", "CONTEXT", "ACTIONS", "OBJECTS"])

def extract_dynamic_from_vlm_text(vlm_text: str) -> str:
    """
    Parses the scene dynamic/micro-narrative from the VLM response.
    """
    return extract_section_content(vlm_text, "DYNAMIC", ["DESCRIPTION", "MOOD", "CONTEXT", "ACTIONS", "OBJECTS"])

def extract_context_from_vlm_text(vlm_text: str) -> str:
    """
    Parses the implied context/intent from the VLM response.
    """
    return extract_section_content(vlm_text, "CONTEXT", ["DESCRIPTION", "MOOD", "DYNAMIC", "ACTIONS", "OBJECTS"])


def extract_objects_from_vlm_text(vlm_text: str) -> List[str]:
    """
    Parses key objects list from the VLM response.
    """
    if not vlm_text:
        return []
    matches = re.findall(r"^\s*[-*•#\d.]*\s*OBJECTS:\s*(.*)$", vlm_text, re.IGNORECASE | re.MULTILINE)
    if not matches:
        return []
    raw_str = matches[-1].strip()
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        elif isinstance(parsed, str):
            return [x.strip() for x in parsed.split(",") if x.strip()]
    except Exception:
        cleaned = re.sub(r'[\[\]"\'`]', '', raw_str)
        if cleaned:
            return [x.strip() for x in cleaned.split(",") if x.strip()]
    return []
