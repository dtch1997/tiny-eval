from pathlib import Path
import json
from typing import Optional

# Add missing imports for type annotations.
from tiny_eval.inference.interface import InferenceAPIInterface
from tiny_eval.inference.types import InferencePrompt, InferenceParams, InferenceResponse

# Define a default cache path if none is provided.
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "tiny_eval" / "inference_cache.json"

def _generate_cache_key(prompt: InferencePrompt, params: InferenceParams) -> str:
    """
    Generate a unique cache key based on the prompt and parameters.
    This function serializes the prompt and parameters to JSON and concatenates them.
    
    Args:
        prompt: The inference prompt.
        params: The inference parameters.
    
    Returns:
        A unique string key.
    """
    key_data = {
        "prompt": prompt.dict(),
        "params": params.dict()
    }
    return json.dumps(key_data, sort_keys=True)

class InferenceCache:
    def __init__(self, api: InferenceAPIInterface, cache_path: Optional[Path] = None) -> None:
        """
        Initialize the InferenceCache.

        Args:
            api: The underlying Inference API to use.
            cache_path: The file path to persist the cache. If None, a default path is used.
        """
        self.api = api
        self.cache_path: Path = cache_path if cache_path is not None else DEFAULT_CACHE_PATH
        self._cache: dict = {}  # initialize internal cache as an empty dictionary

        # If a cache file exists, attempt to load from disk.
        if self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}
        # ... any additional initialization ... 

    async def _get_response(self, prompt: InferencePrompt, params: InferenceParams) -> list[InferenceResponse]:
        """
        Get responses for a given prompt and parameters from the cache if available,
        otherwise call the underlying API and cache the result.

        Args:
            prompt: The inference prompt.
            params: The inference parameters.

        Returns:
            A list of InferenceResponse objects.
        """
        # Generate a cache key as a string based on prompt and parameters using the helper function.
        key_str = _generate_cache_key(prompt, params)
        
        if key_str in self._cache:
            return self._cache[key_str]
        
        responses = await self.api._get_response(prompt, params)
        self._cache[key_str] = responses
        return responses

    def save_cache(self) -> None:
        """
        Save the current cache to disk as a JSON file.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2)) 