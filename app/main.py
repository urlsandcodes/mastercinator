import argparse
import asyncio
import sys
import os
import json
from pathlib import Path
from app.config import config
from app.logger import logger
from app.manager import process_videos
from display import console, json_view

async def run_task_mode(input_path: Path, output_path: Path) -> int:
    """
    Reads tasks from task file, processes them concurrently,
    and writes style captions results.json file.
    """
    logger.info(f"Running in Captioning Agent Mode. Reading from: {input_path}")
    
    try:
        with open(input_path, "r") as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read tasks file {input_path}: {e}")
        return 1
        
    if not isinstance(tasks, list):
        logger.error("Input tasks.json must be a list of task objects.")
        return 1
        
    try:
        # Process the tasks list
        results = await process_videos([], max_parallel=config.max_parallel, tasks=tasks)
        
        # Fail loudly if any task failed
        failed_tasks = [r for r in results if r.status == "failed"]
        if failed_tasks:
            logger.error(f"Task processing failed. {len(failed_tasks)} tasks failed.")
            for ft in failed_tasks:
                logger.error(f"Task {ft.id} failed with errors: {ft.stage_errors}")
            return 1
            
        # Format mapping output schema
        submission_results = []
        for r in results:
            submission_results.append({
                "task_id": r.id,
                "captions": r.captions or {}
            })
            
        # Ensure parent output directory exists (needed for custom local test runs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Atomic write: write to a temp file in the same directory, then rename it
        import tempfile
        
        dir_name = output_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(submission_results, f, ensure_ascii=True, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            
        logger.info(f"Successfully exported final captions to: {output_path}")
        return 0
    except Exception as e:
        logger.exception(f"Execution error during caption tasks processing: {e}")
        return 1

async def main_async(
    videos: list[str], 
    output_json: bool, 
    max_parallel: int, 
    tasks_path: str, 
    results_path: str,
    pipeline_mode: str
) -> int:
    """Async entry wrapper that runs the analysis pipeline for the inputs."""
    config.pipeline_mode = pipeline_mode
    logger.info(f"Using pipeline mode: {config.pipeline_mode}")
    # Database removed

    input_path = Path(tasks_path)
    output_path = Path(results_path)

    # Auto-detect task mode (if file exists)
    if input_path.exists():
        return await run_task_mode(input_path, output_path)
        
    logger.info("Initializing Video Intelligence CLI Prototype...")
    logger.info(f"Configuration: VLM_PROVIDER={config.vlm_provider}, Parallel Limits={max_parallel}")
    
    if not videos:
        logger.error("No input files provided. Please pass video files or remote URLs, or mount tasks file.")
        return 1
        
    try:
        results = await process_videos(videos, max_parallel=max_parallel)
        
        # Display the parsed results
        for r in results:
            if output_json:
                json_view.render_json(r)
            else:
                console.render(r)
                
        # Return success status code if at least one video succeeded
        failed_count = sum(1 for r in results if r.status == "failed")
        if failed_count == len(results):
            logger.error("All video analyses failed.")
            return 1
        elif failed_count > 0:
            logger.warning(f"Analysis completed with {failed_count} failures.")
            return 0
        return 0
    except Exception as e:
        logger.exception(f"Unhandled runtime exception: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Local Video Intelligence CLI Prototype / Video Captioning Agent."
    )
    parser.add_argument(
        "videos",
        nargs="*", # Changed to * so it is optional when tasks.json is auto-detected
        help="Mixed list of local file paths and remote URLs."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Outputs the VideoResult schemas directly as raw JSON objects to stdout."
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=config.max_parallel,
        help="Maximum concurrent video streams to process at once."
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default="/input/tasks.json",
        help="Path to tasks.json (defaults to /input/tasks.json)."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="/output/results.json",
        help="Path to save results.json (defaults to /output/results.json)."
    )
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default=config.pipeline_mode,
        choices=["modular", "qwen_direct"],
        help="Select pipeline to run (modular or qwen_direct). Defaults to qwen_direct."
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = asyncio.run(
            main_async(
                args.videos, 
                args.json, 
                args.parallel, 
                args.tasks_path, 
                args.results_path,
                args.pipeline_mode
            )
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process halted by KeyboardInterrupt.")
        sys.exit(130)

if __name__ == "__main__":
    main()
