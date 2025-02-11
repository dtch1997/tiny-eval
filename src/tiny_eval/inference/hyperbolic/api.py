from openai import AsyncOpenAI
from typing import Optional

from ..interface import InferenceAPIInterface
from tiny_eval.inference.data_models import (
    InferencePrompt,
    InferenceParams,
    InferenceResponse,
)
from tiny_eval.core.constants import HYPERBOLIC_BASE_URL, HYPERBOLIC_API_KEY
from tiny_eval.inference.openai.api import _openai_chat_completion as _hyperbolic_chat_completion
from tiny_eval.inference.utils.rate_limiter import AsyncRateLimiter

class HyperbolicInferenceAPI(InferenceAPIInterface):
    """
    Inference API for Hyperbolic with rate limiting of 600 requests per 60 secondss
    """
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
    ):
        self.client = client or AsyncOpenAI(
            api_key=HYPERBOLIC_API_KEY,
            base_url=HYPERBOLIC_BASE_URL,
        )
    
    # 600 requests per 60 seconds
    @AsyncRateLimiter(requests=10, window=60)
    async def __call__(
        self, 
        model: str,
        prompt: InferencePrompt,
        params: InferenceParams,
    ) -> InferenceResponse:
        """
        Make an inference call to Hyperbolic API with rate limiting
        
        Args:
            model: The model identifier
            prompt: The inference prompt
            params: Inference parameters
            
        Returns:
            InferenceResponse: The model's response
        """

        return await _hyperbolic_chat_completion(
            client=self.client,
            model=model,
            prompt=prompt,
            params=params,
        )