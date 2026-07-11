import json
import httpx
import asyncio
from app.config import config
from app.logger import logger


def clean_special_characters(text: str) -> str:
    """Replaces Unicode em-dashes, en-dashes, and curly quotes with standard ASCII equivalents."""
    if not text:
        return ""
    # Replace em-dashes and en-dashes with a standard hyphen (surrounded by spaces)
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace curly quotes with standard straight quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text

async def summarize_video(transcript: str, timeline_str: str) -> tuple[str, str, list[str]]:
    """
    Calls the Fireworks LLM to summarize the video transcript and event timeline.
    Returns a tuple of (summary, detailed_summary, tags).
    """
    provider = config.vlm_provider
    from llm.prompts import LLM_SUMMARIZE_SYSTEM_PROMPT, LLM_SUMMARIZE_USER_PROMPT
    user_prompt = LLM_SUMMARIZE_USER_PROMPT.format(transcript=transcript or "No transcript available.", timeline=timeline_str)
    
    logger.info(f"Calling LLM summarizer via provider: {provider}")
    if provider == "mock":
        logger.info("Using mock LLM summarizer fallback.")
        summary = "A concise walkthrough showcasing a local video intelligence system in mock execution."
        detailed_summary = (
            "The system executes stages in-memory under concurrency constraints. "
            "It extracts frame representations, runs vision-language prompts to describe scenes, "
            "tracks visual and speech timestamps, and merges everything into a structured report."
        )
        tags = ["prototype", "video-analysis", "architecture", "asyncio"]
        return summary, detailed_summary, tags

    max_retries = 3
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": config.fireworks_llm_model,
                "messages": [
                    {"role": "system", "content": LLM_SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "reasoning_effort": "none"
            }
            headers = {
                "Authorization": f"Bearer {config.fireworks_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json=payload, headers=headers, timeout=60.0
                )
                if resp.status_code == 429:
                    logger.warning(f"Fireworks LLM rate-limited (429). Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                    if attempt == max_retries:
                        resp.raise_for_status()
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]

            # Clean special characters first
            text = clean_special_characters(text)

            # Clean markdown code blocks from the output if present
            text_clean = text.strip()
            if text_clean.startswith("```"):
                lines = text_clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                text_clean = "\n".join(lines).strip()

            data = json.loads(text_clean)
            return (
                data.get("summary", "").strip(),
                data.get("detailed_summary", "").strip(),
                [t.strip() for t in data.get("tags", [])]
            )
        except Exception as e:
            logger.warning(f"LLM summarizer API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2.0
async def synthesize_narrative(description: str, timeline_str: str) -> str:
    """
    Calls the Fireworks LLM to synthesize a deeper narrative interpretation
    from the raw VLM description and timeline events.
    Returns a 2-3 sentence narrative string.
    """
    from llm.prompts import NARRATIVE_SYNTHESIS_SYSTEM_PROMPT, NARRATIVE_SYNTHESIS_USER_PROMPT
    
    provider = config.vlm_provider
    if provider == "mock":
        return "A compelling moment unfolds as subjects interact within their environment, revealing a small but meaningful narrative arc."
    
    user_prompt = NARRATIVE_SYNTHESIS_USER_PROMPT.format(
        description=description or "No visual description available.",
        timeline=timeline_str
    )
    
    max_retries = 3
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": config.fireworks_llm_model,
                "messages": [
                    {"role": "system", "content": NARRATIVE_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200,
                "reasoning_effort": "none"
            }
            headers = {
                "Authorization": f"Bearer {config.fireworks_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json=payload, headers=headers, timeout=30.0
                )
                if resp.status_code == 429:
                    logger.warning(f"Narrative synthesis rate-limited (429). Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                    if attempt == max_retries:
                        resp.raise_for_status()
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                text = clean_special_characters(text)
                return text.strip()
        except Exception as e:
            logger.warning(f"Narrative synthesis API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                # Graceful fallback: use raw description as narrative
                logger.warning("Narrative synthesis failed after all retries. Falling back to raw description.")
                return description or "No narrative could be synthesized."
            await asyncio.sleep(backoff)
            backoff *= 2.0
    return description or "No narrative could be synthesized."

async def generate_single_caption(timeline_str: str, style: str, transcript: str = None, narrative: str = None) -> str:
    """
    Calls the Fireworks LLM for a single style to prevent style bleed.
    Uses static system prompts with strict length/grounding few-shot guidelines.
    Accepts an optional narrative interpretation for deeper caption grounding.
    """
    from llm.prompts import STYLE_SYSTEM_PROMPTS, ZERO_SHOT_STYLE_SYSTEM_PROMPTS, STYLE_USER_PROMPT
    
    if config.prompt_style == "zero_shot":
        system_prompt = ZERO_SHOT_STYLE_SYSTEM_PROMPTS.get(style)
    else:
        system_prompt = STYLE_SYSTEM_PROMPTS.get(style)

    if not system_prompt:
        system_prompt = f"Write a single punchy '{style}' style caption."
        
    user_prompt = STYLE_USER_PROMPT.format(
        timeline=timeline_str,
        narrative=narrative or "No narrative interpretation available.",
        transcript=transcript or "No spoken speech detected in audio."
    )
    
    max_retries = 3
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": config.fireworks_llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7 if style != "formal" else 0.2,
                "max_tokens": 120,
                "reasoning_effort": "none"
            }
            headers = {
                "Authorization": f"Bearer {config.fireworks_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json=payload, headers=headers, timeout=30.0
                )
                if resp.status_code == 429:
                    logger.warning(f"Fireworks LLM {style} rate-limited (429). Attempt {attempt}/{max_retries}. Retrying in {backoff}s...")
                    if attempt == max_retries:
                        resp.raise_for_status()
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                
                text = clean_special_characters(text)
                return text.strip().strip('"').strip()
        except Exception as e:
            logger.warning(f"LLM captioner style {style} API error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2.0
    return f"A video description in {style} style."

async def generate_captions(transcript: str, timeline_str: str, styles: list[str], narrative: str = None) -> dict[str, str]:
    """
    Generates captions in each of the requested styles concurrently, incorporating audio transcript and narrative if available.
    """
    provider = config.vlm_provider

    if provider == "mock":
        logger.info("Using mock LLM captioner fallback.")
        mock_db = {
            "formal": "The video displays a technical prototype execution illustrating async scheduling.",
            "sarcastic": "Oh look, another async Python project that will definitely save the world.",
            "humorous_tech": "When you try to avoid multi-threading in Python so you write 500 lines of asyncio semaphore control.",
            "humorous_non_tech": "A computer screen displaying lots of techy text and coding stuff."
        }
        return {s: mock_db.get(s, "A video caption.") for s in styles}

    logger.info(f"Generating {len(styles)} style captions concurrently via Fireworks ({config.fireworks_llm_model})")
    
    # Run all style generations concurrently to save wall-clock time
    tasks = [generate_single_caption(timeline_str, s, transcript=transcript, narrative=narrative) for s in styles]
    caption_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    result = {}
    for style, caption in zip(styles, caption_results):
        if isinstance(caption, Exception):
            logger.error(f"Failed to generate caption for style {style}: {caption}")
            result[style] = f"A video description in {style} style."
        else:
            result[style] = caption
            
    return result
