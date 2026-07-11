import subprocess
import os
import re
from pathlib import Path
from app.config import config
from app.logger import logger
from media.ffmpeg import FFMPEG_BIN

def detect_scenes(video_path: str, video_id: str, duration: float) -> list[tuple[float, float]]:
    """
    Analyzes scene cuts using the ffmpeg select scene-change filter.
    Returns a list of (start_seconds, end_seconds) intervals representing scenes.
    """
    logger.info(f"Detecting scenes in video: {video_path} (duration={duration:.2f}s)")
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_txt = temp_dir / f"{video_id}_scenes.txt"
    
    # Run ffmpeg with scene selection filter, printing frame info containing timestamps to a temp file
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", video_path,
        "-vf", f"select='gt(scene,0.35)',metadata=print:file={temp_txt}",
        "-f", "null",
        "-"
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        logger.warning(f"Scene detection command failed (perhaps ffmpeg is not fully installed yet): {e}. Defaulting to single scene.")
        return [(0.0, duration if duration > 0 else 10.0)]
        
    timestamps = []
    if temp_txt.exists():
        try:
            with open(temp_txt, "r") as f:
                content = f.read()
            
            # Matches lines like: frame:2 pts:60060 pts_time:2.5025
            pattern = re.compile(r"pts_time:([\d\.]+)")
            for match in pattern.finditer(content):
                ts = float(match.group(1))
                # Ensure we don't double count frame updates within 1s
                if not timestamps or ts - timestamps[-1] > 1.0:
                    timestamps.append(ts)
        except Exception as e:
            logger.error(f"Error parsing scene log: {e}")
        finally:
            # Cleanup temp log file
            try:
                os.remove(temp_txt)
            except Exception:
                pass
                
    # Build scene segments
    scenes = []
    current_start = 0.0
    for ts in timestamps:
        # Avoid tiny scene partitions less than 0.5 seconds
        if ts > current_start + 0.5:
            scenes.append((current_start, ts))
            current_start = ts
            
    # Cap the final segment
    video_dur = duration if duration > 0 else current_start + 5.0
    if video_dur > current_start:
        scenes.append((current_start, video_dur))
        
    # Safeguard: if no scenes found at all, return one full length scene
    if not scenes:
        scenes.append((0.0, video_dur))
        
    logger.info(f"Segmented video into {len(scenes)} scenes: {scenes}")
    return scenes
