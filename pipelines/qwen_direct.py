import base64
import time
import os
import re
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Any, Tuple

from schemas.result import VideoResult
from schemas.vision import Scene
from app.config import config
from app.logger import logger
from media.downloader import download_if_url
from workers.executors import run_in_cpu_pool
from media import ffmpeg
from vision.frames import extract_temporal_frames

def sample_frames(frame_list: List[Dict[str, Any]], num_samples: int = 4) -> List[Dict[str, Any]]:
    """Gets num_samples evenly spaced frames from the list of extracted frames."""
    total = len(frame_list)
    if total <= num_samples:
        return frame_list
    
    selected = []
    for i in range(num_samples):
        # Spread i evenly across the range [0, total-1]
        pos = i * (total - 1) / (num_samples - 1)
        index = round(pos)
        selected.append(frame_list[index])
    return selected

async def generate_qwen_caption(client: httpx.AsyncClient, frames_b64: List[str], style: str) -> str:
    """Sends 4 sampled base64 frames and style prompt directly to Qwen vision model on Fireworks."""
    system_content = (
        "You are a strict data-formatting pipeline. You will receive a persona and an image. "
        "You MUST output a valid JSON object containing exactly one key 'caption' with your final caption as the string value. "
        "Do NOT output any thinking process. Do NOT output any conversational text."
        "Do NOT use Markdown formatting (no asterisks, no headers). Output only raw JSON."
    )
    
    guard = (
        "\n\n### CRITICAL INSTRUCTIONS ###\n"
        "1. You MUST output valid JSON only.\n"
        "2. Your JSON must have a single key 'caption'.\n"
        "3. Do NOT explain your thinking or include any prefix text."
    )
    
    from llm.prompts import STYLE_SYSTEM_PROMPTS, ZERO_SHOT_STYLE_SYSTEM_PROMPTS
    
    if config.prompt_style == "zero_shot":
        prompts = ZERO_SHOT_STYLE_SYSTEM_PROMPTS
    else:
        prompts = STYLE_SYSTEM_PROMPTS
        
    user_prompt_text = prompts.get(style, f"Caption this in a {style} tone.") + guard
    
    # Construct multi-image content payload
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            }
        }
        for img in frames_b64
    ]
    content.append({
        "type": "text",
        "text": user_prompt_text
    })
    
    payload = {
        "model": "accounts/fireworks/models/qwen3p7-plus",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": content}
        ],
        "max_tokens": 400,
        "temperature": 0.7,
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Authorization": f"Bearer {config.fireworks_api_key}",
        "Content-Type": "application/json"
    }
    
    endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    max_retries = 3
    backoff = 2.0
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Pinging Qwen Vision model for style '{style}' (attempt {attempt})")
            resp = await client.post(endpoint, json=payload, headers=headers, timeout=60.0)
            if resp.status_code == 429:
                logger.warning(f"Qwen VLM rate-limited (429). Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2.0
                continue
                
            if resp.status_code != 200:
                logger.error(f"Qwen VLM API Error response body: {resp.text}")
                
            resp.raise_for_status()
            raw_output = resp.json()["choices"][0]["message"]["content"].strip()
            
            import json
            try:
                data = json.loads(raw_output)
                caption = data.get("caption", "").strip()
                if caption:
                    # Clean Unicode special characters
                    from llm.summarize import clean_special_characters
                    caption = clean_special_characters(caption)
                    return caption.strip('"').strip()
            except json.JSONDecodeError:
                logger.warning(f"Qwen VLM output was not valid JSON for style '{style}'. Raw output: {raw_output}")
                return f"[CAPTION_FAILED: '{style}' - malformed JSON output]"
            
            logger.warning(f"Qwen VLM missing 'caption' key for style '{style}'. Raw output: {raw_output}")
            return f"[CAPTION_FAILED: '{style}' - missing caption key]"
        except Exception as e:
            logger.error(f"Qwen VLM API error on attempt {attempt} for style '{style}': {e}")
            if attempt == max_retries:
                return f"[CAPTION FAILED: '{style}' - API ERROR: {e}]"
            await asyncio.sleep(backoff)
            backoff *= 2.0
            
    return f"[CAPTION FAILED: '{style}' - execution failed]"

async def run_qwen_pipeline(video_id: str, video_source: str, styles: List[str] = None) -> VideoResult:
    """
    Runs the simplified single-stage Qwen-VL direct video captioning pipeline.
    Does not run audio extraction or Whisper transcription.
    """
    logger.info(f"Initializing Qwen Direct Pipeline for: {video_source}")
    
    scratch_files: List[str] = []
    timings_ms: Dict[str, int] = {}
    stage_errors: Dict[str, str] = {}
    
    local_video_path = None
    metadata = {}
    duration = 0.0
    status = "ok"
    captions = {}
    
    if not styles:
        styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
        
    try:
        # 1. DOWNLOAD STAGE
        start_time = time.time()
        try:
            local_video_path = await download_if_url(video_source, video_id)
            timings_ms["download"] = int((time.time() - start_time) * 1000)
            if local_video_path.startswith(str(Path(config.temp_dir).resolve())):
                scratch_files.append(local_video_path)
        except Exception as e:
            timings_ms["download"] = int((time.time() - start_time) * 1000)
            logger.error(f"Required download stage failed: {e}")
            raise
            
        # 2. METADATA STAGE
        start_time = time.time()
        try:
            metadata = await run_in_cpu_pool(ffmpeg.extract_metadata, local_video_path)
            timings_ms["extract_metadata"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            timings_ms["extract_metadata"] = int((time.time() - start_time) * 1000)
            logger.error(f"Required metadata extraction stage failed: {e}")
            raise
            
        duration = metadata.get("duration", 0.0)
        
        # 3. TEMPORAL FRAME EXTRACTION STAGE
        frames_extracted = []
        start_time = time.time()
        try:
            frames_extracted = await run_in_cpu_pool(extract_temporal_frames, local_video_path, video_id, duration)
            timings_ms["frame_extract"] = int((time.time() - start_time) * 1000)
            for f in frames_extracted:
                if f.get("path"):
                    scratch_files.append(f["path"])
        except Exception as e:
            status = "partial"
            timings_ms["frame_extract"] = int((time.time() - start_time) * 1000)
            stage_errors["frame_extract"] = str(e)
            logger.warning(f"Temporal frame extraction stage failed: {e}")
            
        # 4. QWEN DIRECT MULTIMODAL INFERENCE STAGE
        if frames_extracted:
            start_time = time.time()
            sampled = sample_frames(frames_extracted, num_samples=4)
            frames_b64 = [base64.b64encode(f["bytes"]).decode("utf-8") for f in sampled]
            
            if config.vlm_provider == "mock" or not config.fireworks_api_key:
                logger.info("Using mock Qwen caption responses.")
                mock_db = {
                    "formal": "The video displays a technical prototype execution illustrating async scheduling.",
                    "sarcastic": "Oh look, another async Python project that will definitely save the world.",
                    "humorous_tech": "When you try to avoid multi-threading in Python so you write 500 lines of asyncio semaphore control.",
                    "humorous_non_tech": "A computer screen displaying lots of techy text and coding stuff."
                }
                captions = {s: mock_db.get(s, f"A video description in {s} style.") for s in styles}
                timings_ms["qwen_inference"] = 10
            else:
                try:
                    async with httpx.AsyncClient() as client:
                        tasks = [generate_qwen_caption(client, frames_b64, s) for s in styles]
                        caption_results = await asyncio.gather(*tasks)
                        
                    for s, cap in zip(styles, caption_results):
                        captions[s] = cap
                    timings_ms["qwen_inference"] = int((time.time() - start_time) * 1000)
                except Exception as e:
                    status = "failed"
                    stage_errors["qwen_inference"] = str(e)
                    logger.error(f"Qwen multimodal inference failed: {e}")
                    raise
        else:
            status = "failed"
            stage_errors["frame_extract"] = "No frames extracted"
            for s in styles:
                captions[s] = f"[CAPTION_FAILED: '{s}' - no frames extracted from video]"
                
    except Exception as e:
        status = "failed"
        stage_errors["pipeline"] = str(e)
        logger.error(f"Qwen pipeline execution failed: {e}")
        
    finally:
        # Cleanup temp/scratch files
        logger.info(f"Initiating scratch space cleanup for Qwen pipeline: {video_id}")
        for path in scratch_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned scratch file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove scratch file {path}: {e}")
                
    # Build timeline of events for DB and schemas (simply mock scene timeline from frames)
    scenes = []
    if frames_extracted:
        for idx, f in enumerate(frames_extracted):
            scenes.append(Scene(
                scene_id=f["scene_id"],
                start=f["timestamp"],
                end=f["timestamp"] + 1.0,
                description=f"Extracted Frame at {f['timestamp']:.2f}s"
            ))
            
    # DB Logging removed
    return VideoResult(
        id=video_id,
        source=video_source,
        duration=duration,
        status=status,
        transcript=None,
        timeline=[],
        summary=captions.get(styles[0], "Video Caption"),
        detailed_summary="\n".join(f"{s}: {cap}" for s, cap in captions.items()),
        tags=styles,
        captions=captions,
        scenes=scenes,
        audio=None,
        stage_errors=stage_errors,
        timings_ms=timings_ms
    )
