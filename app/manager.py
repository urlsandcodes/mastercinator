import asyncio
from typing import List, Dict, Any, Optional
import uuid
from schemas.result import VideoResult
from app.logger import logger, video_id_var
from app.executors import shutdown_executors

async def _worker(
    video_source: str, 
    semaphore: asyncio.Semaphore, 
    video_id: Optional[str] = None, 
    styles: Optional[List[str]] = None
) -> VideoResult:
    # Use specified task_id or generate a unique correlation ID
    vid = video_id or f"vid-{uuid.uuid4().hex[:8]}"
    video_id_var.set(vid)
    
    # Lazy import to prevent circular dependency
    from app.config import config
    if config.pipeline_mode == "qwen_direct":
        from pipelines.qwen_direct import run_qwen_pipeline as run_pipeline
    else:
        from pipelines.modular import run_pipeline
    
    logger.info(f"Queued video source: {video_source} (ID={vid})")
    async with semaphore:
        logger.info(f"Running pipeline ({config.pipeline_mode}) for source: {video_source}")
        try:
            result = await run_pipeline(vid, video_source, styles=styles)
            logger.info(f"Completed pipeline for source: {video_source} (status={result.status})")
            return result
        except Exception as e:
            logger.exception(f"Critical pipeline error for source {video_source}: {e}")
            return VideoResult(
                id=vid,
                source=video_source,
                duration=0.0,
                status="failed",
                stage_errors={"pipeline": str(e)},
                captions={s: f"Failed: {e}" for s in styles} if styles else None
            )

async def process_videos(
    videos: List[str], 
    max_parallel: int = 3, 
    tasks: Optional[List[Dict[str, Any]]] = None
) -> List[VideoResult]:
    """
    Schedules and runs video pipelines concurrently under semaphore constraints.
    Supports standard list of videos, or custom task dicts with styles/ids.
    """
    semaphore = asyncio.Semaphore(max_parallel)
    
    if tasks:
        logger.info(f"Scheduling {len(tasks)} tasks via task file specs with max_parallel={max_parallel}")
        worker_tasks = [
            _worker(
                video_source=t["video_url"], 
                semaphore=semaphore, 
                video_id=t["task_id"], 
                styles=t["styles"]
            ) 
            for t in tasks
        ]
    else:
        logger.info(f"Scheduling {len(videos)} videos with max_parallel={max_parallel}")
        worker_tasks = [_worker(video_source=v, semaphore=semaphore) for v in videos]
        
    results = await asyncio.gather(*worker_tasks)
    
    # Clean up process pools
    shutdown_executors()
    
    return list(results)
