import json
from pathlib import Path
from typing import Optional

from tiny_eval.core.constants import CACHE_DIR
from tiny_eval.core.hashable import HashableBaseModel
from tiny_eval.inference.types import InferenceParams, InferenceResponse, InferencePrompt
from tiny_eval.inference.interface import InferenceAPIInterface

class CacheKey(HashableBaseModel):
    """A key for the inference cache, combining prompt and params"""
    prompt: InferencePrompt
    params: InferenceParams

class InferenceCache(InferenceAPIInterface):
    """A cache wrapper around an inference API interface.
    
    Stores responses for prompt+param combinations to avoid redundant API calls.
    Cache is persisted to disk between runs.
    """
    api: InferenceAPIInterface
    cache_dir: Path
    _cache: dict[CacheKey, list[InferenceResponse]] = {}

    def __init__(
        self,
        api: InferenceAPIInterface,
        *,
        cache_dir: Optional[Path | str] = None,
        cache_path: Optional[Path | str] = None
    ) -> None:
        """
        Initialize the cache, optionally loading from disk.
        
        Args:
            api: The underlying API interface to wrap
            cache_dir: Directory to save/load cache from. Ignored if cache_path is provided.
            cache_path: Full path to the cache file. If provided, it will be used instead of cache_dir.
        """
        self.api = api
        if cache_path is not None:
            self._cache_path = Path(cache_path)
            self.cache_dir = self._cache_path.parent
        else:
            self._cache_path = None
            self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        # Initialize the cache as a plain dictionary.
        self._cache: dict[CacheKey, list[InferenceResponse]] = {}
        self.load_cache()

    @property
    def cache_path(self) -> Path:
        if self._cache_path is not None:
            return self._cache_path
        return self.cache_dir / "inference_cache.json"

    def load_cache(self) -> None:
        """Load the cache from disk if it exists."""
        if not self.cache_path.exists():
            return

        try:
            cache_data: dict[str, list[dict]] = json.loads(self.cache_path.read_text())
            # Convert the string keys back to CacheKey objects
            self._cache = {
                CacheKey.model_validate_json(k): [
                    InferenceResponse.model_validate(r) for r in v
                ]
                for k, v in cache_data.items()
            }
        except Exception as e:
            # If there's any error loading the cache, log it and start fresh
            print(f"Error loading cache from {self.cache_path}: {e}")
            self._cache = {}

    def save_cache(self) -> None:
        """Save the cache to disk."""
        if self.cache_path is None:
            return

        # Create parent directories if they don't exist
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert cache to JSON-serializable format
            cache_data = {
                k.model_dump_json(): [r.model_dump() for r in v]
                for k, v in self._cache.items()
            }
            self.cache_path.write_text(json.dumps(cache_data, indent=2))
        except Exception as e:
            print(f"Error saving cache to {self.cache_path}: {e}")

    async def _get_response(self, prompt: InferencePrompt, params: InferenceParams) -> list[InferenceResponse]:
        """Get responses for a prompt+params combination, using cache if available.
        
        Args:
            prompt: The prompt to get responses for
            params: The parameters to use for inference
            
        Returns:
            A list of inference responses
            
        Note:
            This caches based on both the prompt and params, since different params
            could produce different responses even for the same prompt.
        """
        cache_key = CacheKey(prompt=prompt, params=params)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        responses = await self.api._get_response(prompt, params)
        self._cache[cache_key] = responses
        return responses
