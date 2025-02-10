from openai import AsyncOpenAI
from typing import Optional

from ..interface import InferenceAPIInterface
from tiny_eval.inference.data_models import (
    InferencePrompt,
    InferenceParams,
    InferenceResponse,
)
from tiny_eval.core.constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from tiny_eval.inference.openai.api import _openai_chat_completion as _openrouter_chat_completion
from tiny_eval.inference.utils.rate_limiter import AsyncRateLimiter

class OpenRouterInferenceAPI(InferenceAPIInterface):
    """
    Inference API for OpenRouter with rate limiting of 1500 requests per 10 seconds
    """
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
    ):
        self.client = client or AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    
    # 1400 requests per 10 seconds
    # see ./utils.py for how to check rate limit
    @AsyncRateLimiter(requests=1400, window=10)
    async def __call__(
        self, 
        model: str,
        prompt: InferencePrompt,
        params: InferenceParams,
    ) -> InferenceResponse:
        """
        Make an inference call to OpenRouter API with rate limiting
        
        Args:
            model: The model identifier
            prompt: The inference prompt
            params: Inference parameters
            
        Returns:
            InferenceResponse: The model's response
        """
        if model.startswith("openai/"):
            model = model.replace("openai/", "")

        return await _openrouter_chat_completion(
            client=self.client,
            model=model,
            prompt=prompt,
            params=params,
        )