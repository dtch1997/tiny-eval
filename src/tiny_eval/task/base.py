from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar, Optional, Union, Literal
import json
import logging

from tqdm.asyncio import tqdm_asyncio

# Configure logging
logger = logging.getLogger(__name__)

ConfigT = TypeVar('ConfigT', bound='BaseTaskConfig')
ResultT = TypeVar('ResultT')

class TaskResult(Generic[ResultT]):
    """Class representing the result of a task execution"""
    def __init__(
        self, 
        status: Literal["success", "error"],
        data: Optional[ResultT] = None,
        error: Optional[str] = None
    ):
        self.status = status
        self.data = data
        self.error = error

    def to_dict(self) -> dict:
        """Convert TaskResult to dictionary for JSON serialization"""
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskResult[ResultT]':
        """Create TaskResult from dictionary"""
        return cls(
            status=data.get("status"),
            data=data.get("data"),
            error=data.get("error")
        )

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
    async def run_single(self, config: ConfigT) -> TaskResult[ResultT]:
        """Run a single task with the given configuration"""
        pass
    
    def _get_cache_path(self, config: ConfigT) -> Optional[Path]:
        """Get cache file path for given config"""
        if not self.cache_dir:
            return None
        task_id = config.get_id()
        return self.cache_dir / f"{task_id}.json"
    
    async def run(self, configs: list[ConfigT], desc: str = "Running tasks") -> list[TaskResult[ResultT]]:
        """
        Run multiple task configurations with caching and progress bar
        
        Args:
            configs: List of task configurations to run
            desc: Description for progress bar
            
        Returns:
            List of TaskResults corresponding to input configs
        """
        async def run_with_cache(config: ConfigT) -> TaskResult[ResultT]:
            cache_path = self._get_cache_path(config)
            task_id = config.get_id()
            
            # Try loading from cache
            if cache_path and cache_path.exists():
                try:
                    with open(cache_path) as f:
                        data = json.load(f)
                        result = TaskResult.from_dict(data)
                        if result.status == "success":
                            logger.info(f"Cache hit for task {task_id}")
                            return result
                        else:
                            logger.info(f"Cache hit for task {task_id} but result is error: {result.error}")
                            # delete cache file
                            cache_path.unlink()
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
                        json.dump(result.to_dict(), f)
                except OSError as e:
                    logger.warning(f"Failed to write cache for task {task_id}: {str(e)}")
            
            return result
        
        logger.info(f"Running {len(configs)} tasks with description: {desc}")
        tasks = [run_with_cache(config) for config in configs]
        return await tqdm_asyncio.gather(*tasks, desc=desc) 