import os
from pathlib import Path
from app.config import config
from app.logger import logger
from media.ffmpeg import extract_frame

# Target 10 frames per video for optimal temporal coverage while remaining within token budget
MAX_TEMPORAL_FRAMES = 10
MIN_TEMPORAL_FRAMES = 2
TARGET_FRAME_RESOLUTION = 1024


def extract_temporal_frames(video_path: str, video_id: str, duration: float) -> list[dict]:
    """
    Extracts uniformly-spaced frames across the full video duration for temporal coverage.
    Ensures that boundary frames (near start and near end) are always represented.

    Returns a list of dicts: {scene_id, start, end, timestamp, path, bytes}
    """
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if duration <= 0:
        logger.warning(f"Video duration is {duration}s, cannot extract temporal frames.")
        return []

    # Calculate padding to prevent selecting completely black/empty frames at absolute start/end
    pad = min(0.5, duration * 0.05)
    usable_start = pad
    usable_end = max(duration - pad, pad + 0.1)

    # Scale frame count relative to duration, clamping to target limits
    num_frames = max(MIN_TEMPORAL_FRAMES, min(MAX_TEMPORAL_FRAMES, int(duration / 3) + 1))

    if num_frames == 1:
        timestamps = [usable_start]
    else:
        step = (usable_end - usable_start) / (num_frames - 1)
        timestamps = [usable_start + i * step for i in range(num_frames)]

    logger.info(f"Temporal sampling: {num_frames} frames planned at timestamps {[f'{t:.2f}s' for t in timestamps]} (duration={duration:.2f}s)")

    extracted_frames = []

    for idx, ts in enumerate(timestamps):
        filename = f"{video_id}_temporal_{idx}_at_{ts:.2f}.jpg"
        frame_path = temp_dir / filename

        # Extract frame with native ffmpeg downscaling to 1024px
        success = extract_frame(video_path, ts, str(frame_path), longest_side=TARGET_FRAME_RESOLUTION)
        if success:
            try:
                with open(frame_path, "rb") as f:
                    img_bytes = f.read()

                extracted_frames.append({
                    "scene_id": idx,
                    "start": 0.0,
                    "end": duration,
                    "timestamp": ts,
                    "path": str(frame_path.resolve()),
                    "bytes": img_bytes
                })
            except Exception as e:
                logger.error(f"Could not read extracted frame file {frame_path}: {e}")
        else:
            logger.warning(f"Failed to extract temporal frame {idx} at timestamp {ts:.2f}s")

    logger.info(f"Extracted {len(extracted_frames)} temporal frames out of {num_frames} planned.")
    return extracted_frames


def select_and_extract_frames(video_path: str, video_id: str, scenes: list[tuple[float, float]]) -> list[dict]:
    """
    Legacy: Selects midpoint timestamp of each scene.
    Kept for fallback compatibility.
    """
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    extracted_frames = []

    for idx, (start, end) in enumerate(scenes):
        midpoint = start + (end - start) / 2

        filename = f"{video_id}_scene_{idx}_at_{midpoint:.2f}.jpg"
        frame_path = temp_dir / filename

        success = extract_frame(video_path, midpoint, str(frame_path), longest_side=TARGET_FRAME_RESOLUTION)
        if success:
            try:
                with open(frame_path, "rb") as f:
                    img_bytes = f.read()

                extracted_frames.append({
                    "scene_id": idx,
                    "start": start,
                    "end": end,
                    "timestamp": midpoint,
                    "path": str(frame_path.resolve()),
                    "bytes": img_bytes
                })
            except Exception as e:
                logger.error(f"Could not read extracted frame file {frame_path}: {e}")
        else:
            logger.warning(f"Failed to extract representative frame for scene {idx} at timestamp {midpoint:.2f}s")

    logger.info(f"Extracted {len(extracted_frames)} representative frame images out of {len(scenes)} scenes.")
    return extracted_frames
