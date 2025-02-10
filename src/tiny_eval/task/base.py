from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, Optional
import json
import logging

from tqdm.asyncio import tqdm_asyncio

# Configure logging
logger = logging.getLogger(__name__)

ConfigT = TypeVar('ConfigT', bound='BaseTaskConfig')
ResultT = TypeVar('ResultT')

class BaseTaskConfig(ABC):
    """Base class for task configurations"""
    
    @abstractmethod
    def get_id(self) -> str:
        """Return unique identifier for this task configuration"""
        pass

class Task(Generic[ConfigT, ResultT], ABC):
    """Base class for defining tasks"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: Directory to store task results. If None, caching is disabled.
        """
        self.cache_dir = cache_dir
        if cache_dir:
            logger.info(f"Initializing task cache at {cache_dir}")
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    async def run_single(self, config: ConfigT) -> ResultT:
        """Run a single task with the given configuration"""
        pass
    
    def _get_cache_path(self, config: ConfigT) -> Optional[Path]:
        """Get cache file path for given config"""
        if not self.cache_dir:
            return None
        task_id = config.get_id()
        return self.cache_dir / f"{task_id}.json"
    
    async def run(self, configs: list[ConfigT], desc: str = "Running tasks") -> list[ResultT]:
        """
        Run multiple task configurations with caching and progress bar
        
        Args:
            configs: List of task configurations to run
            desc: Description for progress bar
            
        Returns:
            List of results corresponding to input configs
        """
        async def run_with_cache(config: ConfigT) -> ResultT:
            cache_path = self._get_cache_path(config)
            task_id = config.get_id()
            
            # Try loading from cache
            if cache_path and cache_path.exists():
                try:
                    with open(cache_path) as f:
                        result = json.load(f)
                        logger.info(f"Cache hit for task {task_id}")
                        return result
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted cache file for task {task_id}, will rerun task")
                except OSError as e:
                    logger.warning(f"Failed to read cache for task {task_id}: {str(e)}")
            
            # Run task
            result = await self.run_single(config)
            
            # Cache result
            if cache_path:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(result, f)
                except OSError as e:
                    logger.warning(f"Failed to write cache for task {task_id}: {str(e)}")
            
            return result
        
        logger.info(f"Running {len(configs)} tasks with description: {desc}")
        tasks = [run_with_cache(config) for config in configs]
        return await tqdm_asyncio.gather(*tasks, desc=desc) 