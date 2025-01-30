import asyncio
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from tqdm import tqdm

from .constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY

class ModelAPI(ABC):
    """
    Abstract class for a model API.
    """

    @abstractmethod
    async def get_response(self, prompt: str, model: str) -> str:
        pass

    async def get_response_batch(self, prompts: list[str], model: str, show_progress: bool = True) -> list[str]:
        """Get responses for multiple prompts in parallel"""
        tasks = [self.get_response(prompt, model) for prompt in prompts]
        tasks = tqdm(tasks, disable=not show_progress)
        return await asyncio.gather(*tasks)

class OpenRouterModelAPI(ModelAPI):
    """
    Model API for OpenRouter.
    """
    client: AsyncOpenAI

    def __init__(
        self, client: AsyncOpenAI | None = None
    ):
        self.client = client or AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

    async def get_response(self, prompt: str, model: str) -> str:
        completion = await self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content