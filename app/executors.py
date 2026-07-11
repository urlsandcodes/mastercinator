import asyncio
from concurrent.futures import ProcessPoolExecutor
from app.config import config
from app.logger import logger

cpu_executor = None

def get_cpu_executor() -> ProcessPoolExecutor:
    global cpu_executor
    if cpu_executor is None:
        logger.info(f"Initializing CPU ProcessPoolExecutor with pool size: {config.cpu_pool_size}")
        cpu_executor = ProcessPoolExecutor(max_workers=config.cpu_pool_size)
    return cpu_executor

def shutdown_executors() -> None:
    global cpu_executor
    if cpu_executor is not None:
        logger.info("Shutting down CPU ProcessPoolExecutor")
        cpu_executor.shutdown(wait=True)
        cpu_executor = None

async def run_in_cpu_pool(func, *args, **kwargs):
    """Utility to run sync functions inside the CPU ProcessPoolExecutor, supporting kwargs."""
    loop = asyncio.get_running_loop()
    executor = get_cpu_executor()
    if kwargs:
        def wrapper():
            return func(*args, **kwargs)
        return await loop.run_in_executor(executor, wrapper)
    return await loop.run_in_executor(executor, func, *args)
