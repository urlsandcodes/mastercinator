import time
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple

from schemas.result import VideoResult
from schemas.audio import AudioAnalysis
from schemas.vision import Scene
from app.config import config
from app.logger import logger

# Import stage workers
from media.downloader import download_if_url
from media import ffmpeg
from workers.executors import run_in_cpu_pool
from vision.scenes import detect_scenes
from vision.frames import select_and_extract_frames, extract_temporal_frames
from vision.vlm import get_vlm_client

from vision.ocr import extract_ocr_from_vlm_text
from vision.actions import (
    extract_actions_from_vlm_text,
    extract_description_from_vlm_text,
    extract_objects_from_vlm_text,
    extract_mood_from_vlm_text,
    extract_dynamic_from_vlm_text,
    extract_context_from_vlm_text
)
from fusion.timeline import build_timeline
from llm.summarize import summarize_video, generate_captions, synthesize_narrative
from llm.prompts import VLM_FRAME_PROMPT, VLM_SEQUENCE_PROMPT

async def run_pipeline(video_id: str, video_source: str, styles: List[str] = None) -> VideoResult:
    """
    Orchestrates the modular analysis pipeline stages for a single video.
    Tracks performance, supports graceful degradation, and cleans up scratch files.
    """
    logger.info(f"Initializing pipeline execution for: {video_source}")
    
    # Track scratch files to delete upon completion
    scratch_files: List[str] = []
    
    # Store performance timings and stage errors
    timings_ms: Dict[str, int] = {}
    stage_errors: Dict[str, str] = {}
    
    # Initialize variables for stages
    local_video_path = None
    metadata = {}
    audio_analysis = None
    audio_segments = []
    scenes = []
    scene_objects: List[Scene] = []
    timeline = []
    summary = None
    detailed_summary = None
    tags = []
    status = "ok"
    
    try:
        # 1. DOWNLOAD STAGE (Required)
        start_time = time.time()
        try:
            local_video_path = await download_if_url(video_source, video_id)
            timings_ms["download"] = int((time.time() - start_time) * 1000)
            
            # If downloaded file resides in our temp directory, register it for deletion
            if local_video_path.startswith(str(Path(config.temp_dir).resolve())):
                scratch_files.append(local_video_path)
        except Exception as e:
            timings_ms["download"] = int((time.time() - start_time) * 1000)
            logger.error(f"Required download stage failed: {e}")
            raise
            
        # 2. METADATA STAGE (Required)
        start_time = time.time()
        try:
            metadata = await run_in_cpu_pool(ffmpeg.extract_metadata, local_video_path)
            timings_ms["extract_metadata"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            timings_ms["extract_metadata"] = int((time.time() - start_time) * 1000)
            logger.error(f"Required metadata extraction stage failed: {e}")
            raise
            
        duration = metadata.get("duration", 0.0)
        has_audio = metadata.get("has_audio", False)
        
        # Audio transcription disabled/removed from pipeline as it is not needed.
        audio_analysis = None
        audio_segments = []
                
        # 5. TEMPORAL FRAME EXTRACTION STAGE
        # Extract uniformly-spaced frames directly using duration-based sampling.
        frames_extracted = []
        start_time = time.time()
        try:
            frames_extracted = await run_in_cpu_pool(extract_temporal_frames, local_video_path, video_id, duration)
            timings_ms["frame_extract"] = int((time.time() - start_time) * 1000)
            # Register frame files for cleanup
            for f in frames_extracted:
                if f.get("path"):
                    scratch_files.append(f["path"])
        except Exception as e:
            status = "partial"
            timings_ms["frame_extract"] = int((time.time() - start_time) * 1000)
            stage_errors["frame_extract"] = str(e)
            logger.warning(f"Temporal frame extraction stage failed: {e}")
                
        # 6. VLM SEQUENCE DESCRIPTION STAGE
        # Send all frames in a single multi-image API call for temporal-aware description.
        desc = ""
        vlm_narrative_input = ""
        inference_type = "scenes"
        if frames_extracted:
            start_time = time.time()
            try:
                vlm_client = get_vlm_client()
                images = [f["bytes"] for f in frames_extracted]
                
                # Construct chronological frame timing manifest to pass to the VLM
                manifest_lines = []
                for idx, f in enumerate(frames_extracted):
                    manifest_lines.append(f"Frame {idx + 1}: timestamp {f['timestamp']:.2f}s")
                manifest_text = "Chronological Frame Sequence Metadata:\n" + "\n".join(manifest_lines)
                
                # Use multi-image sequence call with timing metadata for temporal alignment
                raw_desc = await vlm_client.describe_frames_sequence(images, VLM_SEQUENCE_PROMPT, manifest_text=manifest_text)
                timings_ms["vlm"] = int((time.time() - start_time) * 1000)
                vlm_success = True
                
                # Parse the unified description into a single Scene object covering the whole video
                desc = extract_description_from_vlm_text(raw_desc)
                mood = extract_mood_from_vlm_text(raw_desc)
                narrative_arc = extract_dynamic_from_vlm_text(raw_desc)
                implied_context = extract_context_from_vlm_text(raw_desc)
                
                vlm_narrative_input = f"Literal description: {desc}\nMood: {mood}\nNarrative Arc: {narrative_arc}\nContext: {implied_context}"
                
                ocr = [] # extract_ocr_from_vlm_text(raw_desc)
                actions = extract_actions_from_vlm_text(raw_desc)
                objects = extract_objects_from_vlm_text(raw_desc)
                
                scene_objects.append(Scene(
                    scene_id=0,
                    start=0.0,
                    end=duration,
                    description=desc,
                    objects=objects,
                    actions=actions,
                    ocr=ocr
                ))

                # Log the unified temporal description
                table_lines = [
                    "\n==========================================================================",
                    f"TEMRAPOL VLM DESCRIPTION ({len(images)} frames analyzed in single call)",
                    "==========================================================================",
                    f"+---------+---------+---------+--------------------------------------------------------+",
                    f"| Frames  | Start   | End     | VLM Temporal Description & Details                     |",
                    f"+---------+---------+---------+--------------------------------------------------------+"
                ]
                desc_snippet = desc[:52] + "..." if len(desc) > 52 else desc.ljust(55)
                table_lines.append(f"| {str(len(images)).ljust(7)} | {'0.00s'.ljust(7)} | {f'{duration:.2f}s'.ljust(7)} | {desc_snippet.ljust(54)} |")
                if ocr:
                    table_lines.append(f"|         |         |         |   OCR Detected: {str(ocr)[:50].ljust(50)} |")
                if actions:
                    table_lines.append(f"|         |         |         |   Actions: {str(actions)[:52].ljust(52)} |")
                table_lines.append(f"+---------+---------+---------+--------------------------------------------------------+")
                table_lines.append("==========================================================================\n")
                logger.info("\n".join(table_lines))

            except Exception as e:
                status = "partial"
                timings_ms["vlm"] = int((time.time() - start_time) * 1000)
                stage_errors["vlm"] = str(e)
                logger.warning(f"VLM sequence processing stage failed: {e}. Falling back to per-frame descriptions.")
                
                # Fallback: describe each frame independently using the single-frame prompt
                try:
                    start_time_fb = time.time()
                    vlm_client = get_vlm_client()
                    images = [f["bytes"] for f in frames_extracted]
                    descriptions = await vlm_client.describe_frames_batch(images, VLM_FRAME_PROMPT)
                    
                    for frame, raw_desc_fb in zip(frames_extracted, descriptions):
                        desc_fb = extract_description_from_vlm_text(raw_desc_fb)
                        ocr_fb = [] #extract_ocr_from_vlm_text(raw_desc_fb)
                        actions_fb = extract_actions_from_vlm_text(raw_desc_fb)
                        objects_fb = extract_objects_from_vlm_text(raw_desc_fb)
                        
                        scene_objects.append(Scene(
                            scene_id=frame["scene_id"],
                            start=frame["start"],
                            end=frame["end"],
                            description=desc_fb,
                            objects=objects_fb,
                            actions=actions_fb,
                            ocr=ocr_fb
                        ))
                    timings_ms["vlm_fallback"] = int((time.time() - start_time_fb) * 1000)
                    desc = " ".join([sc.description for sc in scene_objects])
                    vlm_narrative_input = desc
                    logger.info(f"VLM fallback completed: described {len(descriptions)} frames individually.")
                except Exception as e2:
                    stage_errors["vlm_fallback"] = str(e2)
                    logger.warning(f"VLM fallback also failed: {e2}")
                
        # 8. TIMELINE MERGE STAGE (Optional)
        start_time = time.time()
        try:
            timeline = build_timeline(audio_segments, scene_objects)
            timings_ms["timeline"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            status = "partial"
            timings_ms["timeline"] = int((time.time() - start_time) * 1000)
            stage_errors["timeline"] = str(e)
            logger.warning(f"Timeline merge stage failed: {e}")
            
        # 9. LLM SUMMARIZATION / CAPTIONS STAGE (Optional)
        start_time = time.time()
        captions = None
        try:
            # Format timeline events as string for the LLM summarizer context
            timeline_str = ""
            for ev in timeline:
                timeline_str += f"[{ev.time_display}] ({ev.source}) {ev.event}\n"
                
            transcript_text = audio_analysis.transcript if audio_analysis else None

            # Log how we knit the information together for context
            knitting_lines = [
                "\n==========================================================================",
                "KNITTING SCENE & AUDIO TIMELINE CONTEXT FOR GENERATION",
                "==========================================================================",
                f"Transcript Text: {transcript_text or '<No Audio/Speech Found>'}",
                f"Timeline Event Sequence:\n" + "\n".join(f"  - [{ev.time_display}] ({ev.source}) {ev.event}" for ev in timeline),
                "==========================================================================\n"
            ]
            logger.info("\n".join(knitting_lines))
            
            # 9b. NARRATIVE SYNTHESIS STAGE — extract deeper meaning before styling
            narrative = None
            try:
                start_narrative = time.time()
                narrative = await synthesize_narrative(vlm_narrative_input or desc, timeline_str)
                timings_ms["narrative_synthesis"] = int((time.time() - start_narrative) * 1000)
                
                # Log the synthesized narrative
                narrative_lines = [
                    "\n==========================================================================",
                    "NARRATIVE SYNTHESIS (deeper meaning extracted)",
                    "==========================================================================",
                    narrative,
                    "==========================================================================\n"
                ]
                logger.info("\n".join(narrative_lines))
            except Exception as e:
                timings_ms["narrative_synthesis"] = int((time.time() - start_narrative) * 1000)
                stage_errors["narrative_synthesis"] = str(e)
                logger.warning(f"Narrative synthesis stage failed: {e}. Captions will proceed without narrative.")

            if not styles:
                styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
                
            captions = await generate_captions(transcript_text, timeline_str, styles, narrative=narrative)
            summary = captions.get(styles[0], "Video Caption")
            detailed_summary = "\n".join(f"{s}: {cap}" for s, cap in captions.items())
            tags = styles

            # Log final captions by tone
            caption_lines = [
                "\n==========================================================================",
                "FINAL GENERATED CAPTIONS BY TONE",
                "=========================================================================="
            ]
            for tone, cap in captions.items():
                caption_lines.append(f"[{tone.upper()}]: {cap}")
            caption_lines.append("==========================================================================\n")
            logger.info("\n".join(caption_lines))
                
            timings_ms["summarize"] = int((time.time() - start_time) * 1000)
        except Exception as e:
            if styles:
                raise
            status = "partial"
            timings_ms["summarize"] = int((time.time() - start_time) * 1000)
            stage_errors["summarize"] = str(e)
            logger.warning(f"LLM summarizer stage failed: {e}")
            
    except Exception as e:
        status = "failed"
        stage_errors["pipeline"] = str(e)
        logger.error(f"Video pipeline execution failed critically: {e}")
        
    finally:
        # Clean up scratch files
        logger.info(f"Initiating scratch space cleanup for pipeline: {video_id}")
        for path in scratch_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned scratch file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove scratch file {path}: {e}")
                
    # DB Logging removed

    # Build and return the final VideoResult object
    return VideoResult(
        id=video_id,
        source=video_source,
        duration=duration if metadata else 0.0,
        status=status,
        transcript=audio_analysis.transcript if audio_analysis else None,
        timeline=timeline,
        summary=summary,
        detailed_summary=detailed_summary,
        tags=tags,
        captions=captions,
        scenes=scene_objects,
        audio=audio_analysis,
        stage_errors=stage_errors,
        timings_ms=timings_ms
    )
