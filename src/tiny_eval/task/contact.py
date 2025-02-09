from dataclasses import dataclass
from typing import Dict, Any
import hashlib

from .base import Task, BaseTaskConfig
from tiny_eval.core.constants import Model

@dataclass
class ContactTaskConfig(BaseTaskConfig):
    """Configuration for Contact task"""
    alice: Model
    bob: Model
    dean: Model
    secret_word: str
    max_turns: int = 10
    name: str | None = None
    
    def get_id(self) -> str:
        """Generate unique ID based on config parameters"""
        if self.name:
            return self.name
            
        # Create deterministic hash of config parameters
        config_str = f"{self.alice.value}_{self.bob.value}_{self.dean.value}_{self.secret_word}_{self.max_turns}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

class ContactTask(Task[ContactTaskConfig, Dict[str, Any]]):
    """Implementation of the Contact game task"""
    
    async def run_single(self, config: ContactTaskConfig) -> Dict[str, Any]:
        # Existing run_task implementation from task.py would go here
        pass 