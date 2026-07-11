import subprocess
import shutil
import json
import os
from pathlib import Path
from app.logger import logger

def _find_binary(name: str) -> str:
    """Finds a system binary in PATH or common Homebrew/Mac directories."""
    path = shutil.which(name)
    if path:
        return path
    for p in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]:
        full_path = os.path.join(p, name)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            return full_path
    return name

FFMPEG_BIN = _find_binary("ffmpeg")
FFPROBE_BIN = _find_binary("ffprobe")

def extract_metadata(video_path: str) -> dict:
    """
    Parses video stream details via ffprobe.
    Returns details on duration, resolution, and presence of audio.
    """
    logger.info(f"Extracting metadata for: {video_path}")
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        
        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0.0))
        
        width = 0
        height = 0
        has_audio = False
        
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type")
            if codec_type == "video":
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
            elif codec_type == "audio":
                has_audio = True
                
        metadata = {
            "duration": duration,
            "width": width,
            "height": height,
            "has_audio": has_audio,
            "raw": data
        }
        logger.info(f"Extracted metadata: {width}x{height}, duration={duration:.2f}s, has_audio={has_audio}")
        return metadata
    except Exception as e:
        logger.error(f"ffprobe execution failed for {video_path}: {e}")
        raise IOError(f"Could not parse metadata using ffprobe: {e}")

def extract_audio(video_path: str, audio_output_path: str) -> str:
    """
    Extracts the audio track to mono 16kHz MP3 format.
    """
    logger.info(f"Extracting audio track: {video_path} -> {audio_output_path}")
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ac", "1",
        "-ar", "16000",
        "-q:a", "4",
        audio_output_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully extracted audio: {audio_output_path}")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg audio extraction failed: {e.stderr}")
        raise IOError(f"Failed to extract audio track: {e.stderr}")

def extract_frame(video_path: str, timestamp: float, output_path: str, longest_side: int = 1024) -> bool:
    """
    Extracts a single video frame at the given timestamp as a JPEG image,
    natively downscaling it to ensure the longest side does not exceed longest_side.
    """
    logger.info(f"Extracting frame at {timestamp:.2f}s (scaled to {longest_side}px) -> {output_path}")
    
    # Scale filter: sets longest side to longest_side, preserves aspect ratio, ensures even dimensions
    scale_filter = f"scale=w='if(gt(iw,ih),{longest_side},-2)':h='if(gt(iw,ih),-2,{longest_side})'"
    
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-vf", scale_filter,
        "-vframes", "1",
        "-q:v", "5",  # Balance quality vs payload (~80% quality)
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg frame extraction failed at {timestamp:.2f}s: {e.stderr}")
        return False
