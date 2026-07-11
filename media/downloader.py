import os
import httpx
import asyncio
from pathlib import Path
from app.config import config
from app.logger import logger

async def download_if_url(video_source: str, video_id: str) -> str:
    """
    Checks if a video source is a URL or a local path.
    Downloads remote videos to scratch space with retries and size limits.
    Returns the resolved path to the local video file.
    """
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if URL
    if video_source.startswith(("http://", "https://")):
        logger.info(f"Source is a remote URL. Commencing download: {video_source}")
        
        # Clean filename extraction
        parsed_name = os.path.basename(video_source.split('?')[0])
        if not parsed_name or len(parsed_name) < 3:
            parsed_name = "downloaded_video.mp4"
        filename = f"{video_id}_{parsed_name}"
        dest_path = temp_dir / filename
        
        max_retries = 3
        backoff = 1.0
        
        for attempt in range(1, max_retries + 1):
            try:
                # Use client timeout from config
                timeout_sec = config.stage_timeouts.get("download", 120)
                async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
                    async with client.stream("GET", video_source) as response:
                        response.raise_for_status()
                        
                        cl = response.headers.get("content-length")
                        if cl:
                            size_mb = int(cl) / (1024 * 1024)
                            if size_mb > config.max_download_mb:
                                raise ValueError(f"Content-Length ({size_mb:.1f} MB) exceeds maximum download size ({config.max_download_mb} MB)")
                        
                        bytes_downloaded = 0
                        with open(dest_path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                bytes_downloaded += len(chunk)
                                if bytes_downloaded / (1024 * 1024) > config.max_download_mb:
                                    raise ValueError(f"Downloaded bytes exceeded maximum limit ({config.max_download_mb} MB)")
                                f.write(chunk)
                                
                        logger.info(f"Successfully downloaded video to {dest_path} ({bytes_downloaded / (1024*1024):.1f} MB)")
                        return str(dest_path.resolve())
            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed for {video_source}: {e}")
                if attempt == max_retries:
                    raise IOError(f"Failed to download remote file after {max_retries} attempts: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2.0
    else:
        # Local file
        path = Path(video_source)
        if not path.is_file():
            raise FileNotFoundError(f"Local file does not exist: {video_source}")
        logger.info(f"Source verified as local file: {path.resolve()}")
        return str(path.resolve())
