from typing import Tuple, List, Dict, Any
from schemas.audio import AudioAnalysis
from audio.whisper import transcribe_audio
from audio.music import detect_music
from audio.ambient import detect_ambient
from app.logger import logger

async def analyze_audio(audio_path: str) -> Tuple[AudioAnalysis, List[Dict[str, Any]]]:
    """
    Orchestrates the audio transcription stage and runs secondary sound detectors.
    Returns the AudioAnalysis schema and the raw timed segments for timeline integration.
    """
    logger.info(f"Starting audio analysis pipeline for: {audio_path}")
    errors = []
    transcript = None
    speech = False
    segments = []
    
    try:
        raw_response = await transcribe_audio(audio_path)
        transcript = raw_response.get("text", "").strip()
        speech = len(transcript) > 0
        segments = raw_response.get("segments", [])
    except Exception as e:
        logger.error(f"Audio analysis transcription failed: {e}")
        errors.append(f"transcription_stage_error: {e}")
        
    music_present = False
    music_mood = None
    ambient_sounds = []
    
    # Run secondary classifiers if we have transcript content
    if transcript:
        try:
            music_present, music_mood = detect_music(transcript)
            ambient_sounds = detect_ambient(transcript)
        except Exception as e:
            logger.error(f"Audio heuristics execution failed: {e}")
            errors.append(f"heuristics_stage_error: {e}")
            
    analysis = AudioAnalysis(
        speech=speech,
        transcript=transcript,
        music=music_present,
        music_mood=music_mood,
        ambient=ambient_sounds,
        errors=errors
    )
    
    return analysis, segments
