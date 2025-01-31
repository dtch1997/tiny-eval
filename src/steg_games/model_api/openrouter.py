from openai import AsyncOpenAI


from steg_games.core.constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from .base import ModelAPIInterface
from steg_games.core.types import Message, Choice

class OpenRouterModelAPI(ModelAPIInterface):
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

    async def get_response(
        self, 
        messages: list[Message], 
        model: str,
        # extra options 
        n: int = 1,
        temperature: float = 1.0,
        logprobs: bool = False
    ) -> list[Choice]:
        completion = await self.client.chat.completions.create(
            model=model,
            messages=[message.to_dict() for message in messages],
            n=n,
            temperature=temperature,
            logprobs=logprobs,
        )
        return completion.choices
