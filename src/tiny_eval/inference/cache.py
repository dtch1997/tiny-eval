import json
from pathlib import Path
from typing import Optional

from tiny_eval.core.constants import CACHE_DIR
from tiny_eval.core.hashable import HashableBaseModel
from tiny_eval.inference.types import InferenceParams, InferenceResponse, InferencePrompt, InferenceChoice
from tiny_eval.inference.interface import InferenceAPIInterface

class CacheKey(HashableBaseModel):
    """A key for the inference cache, combining prompt and params"""
    prompt: InferencePrompt
    params: InferenceParams

    def __hash__(self) -> int:
        """Custom hash implementation to handle unhashable types."""
        # Convert messages list to tuple to make it hashable
        messages_tuple = tuple(
            (msg.role, msg.content) for msg in self.prompt.messages
        )
        # Create hash from the hashable components
        return hash((
            messages_tuple,
            self.params.model_dump_json()  # Convert params to stable string representation
        ))

    def __eq__(self, other: object) -> bool:
        """Custom equality check to match hash behavior."""
        if not isinstance(other, CacheKey):
            return NotImplemented
        return (
            self.prompt.model_dump_json() == other.prompt.model_dump_json() and
            self.params.model_dump_json() == other.params.model_dump_json()
        )

class InferenceCache(InferenceAPIInterface):
    """A cache wrapper around an inference API interface.
    
    Stores responses for prompt+param combinations to avoid redundant API calls.
    Cache is persisted to disk between runs.
    """
    api: InferenceAPIInterface
    cache_dir: Path
    _cache: dict[CacheKey, InferenceResponse] = {}

    def __init__(
        self,
        api: InferenceAPIInterface,
        *,
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        """
        Initialize the cache, optionally loading from disk.
        
        Args:
            api: The underlying API interface to wrap
            cache_dir: Directory to save/load cache from
        """
        self.api = api
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize the cache as a plain dictionary
        self._cache: dict[CacheKey, InferenceResponse] = {}
        self.load_cache()

    @property
    def cache_path(self) -> Path:
        """Path to the cache file within the cache directory."""
        return self.cache_dir / "inference_cache.json"

    def load_cache(self) -> None:
        """Load the cache from disk if it exists."""
        if not self.cache_path.exists():
            return

        try:
            cache_data: dict[str, dict] = json.loads(self.cache_path.read_text())
            # Convert the string keys back to CacheKey objects and deserialize responses
            self._cache = {
                CacheKey.model_validate_json(k): InferenceResponse.model_validate(v)
                for k, v in cache_data.items()
            }
        except Exception as e:
            # If there's any error loading the cache, log it and start fresh
            print(f"Error loading cache from {self.cache_path}: {e}")
            self._cache = {}

    def save_cache(self) -> None:
        """Save the cache to disk."""
        # Create parent directories if they don't exist
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            k.model_dump_json(): v.model_dump()
            for k, v in self._cache.items()
        }
        self.cache_path.write_text(json.dumps(cache_data, indent=2))

    async def _get_response(self, prompt: InferencePrompt, params: InferenceParams) -> InferenceResponse:
        """Get response for a prompt+params combination, using cache if available.
        
        Args:
            prompt: The prompt to get response for
            params: The parameters to use for inference
            
        Returns:
            An inference response
            
        Note:
            This caches based on both the prompt and params, since different params
            could produce different responses even for the same prompt.
        """
        cache_key = CacheKey(prompt=prompt, params=params)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        response = await self.api._get_response(prompt, params)
        self._cache[cache_key] = response
        return response
