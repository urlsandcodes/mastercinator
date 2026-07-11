import httpx
import os
from typing import Dict, Any
from app.config import config
from app.logger import logger

async def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """Transcribes audio using the Fireworks Whisper API."""
    
    if config.whisper_model == "mock":
        logger.info("Using mock transcription client.")
        return {
            "text": "Hello, welcome to the demonstration of the Local Video Intelligence Prototype. In this video, we will walk through the pipeline.",
            "segments": [
                {
                    "start": 1.2,
                    "end": 5.4,
                    "text": "Hello, welcome to the demonstration of the Local Video Intelligence Prototype."
                },
                {
                    "start": 5.8,
                    "end": 9.5,
                    "text": "In this video, we will walk through the pipeline."
                }
            ]
        }

    endpoint = "https://api.fireworks.ai/inference/v1/audio/transcriptions"
    model = config.whisper_model or "whisper-large-v3"
    api_key = config.fireworks_api_key

    logger.info(f"Calling Fireworks Whisper API ({model}) to transcribe: {audio_path}")
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            with open(audio_path, "rb") as f:
                files = {
                    "file": (os.path.basename(audio_path), f, "audio/mpeg")
                }
                data = {
                    "model": model,
                    "response_format": "verbose_json"
                }
                
                resp = await client.post(
                    endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=config.stage_timeouts.get("transcribe", 300)
                )
                resp.raise_for_status()
                result = resp.json()
                logger.info("Successfully transcribed audio with Fireworks Whisper.")
                return result
    except Exception as e:
        logger.error(f"Fireworks Whisper API call failed: {e}.")
        raise
