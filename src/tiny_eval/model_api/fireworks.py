import backoff

from typing import Any
from fireworks.client import ChatCompletion
from fireworks.client.error import (
    RateLimitError,
    InternalServerError,
)

from .base import ModelAPIInterface
from tiny_eval.core._types import Message, Choice

def on_backoff(details):
    """Print exception details on backoff, except for connection errors."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        RateLimitError,
        InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=on_backoff,
)
async def _fireworks_chat_completion(**kwargs) -> Any:
    """Make a request to the Fireworks chat completion API with exponential backoff."""
    return await ChatCompletion.acreate(**kwargs)

class FireworksModelAPI(ModelAPIInterface):
    """Model API for Fireworks.ai."""
    
    def __init__(
        self,
        model: str,
        client: Any | None = None,  # client param kept for compatibility
    ):
        """Initialize the Fireworks API client.
        
        Args:
            model: Name of the model to use
            client: Unused parameter kept for interface compatibility
        """
        self.model = model

    async def _get_response(
        self,
        messages: list[Message],
        *,
        n: int = 1,
        temperature: float = 1.0,
        logprobs: bool = False,
    ) -> str:
        """Get a response from the model.
        
        Args:
            messages: List of messages to send
            n: Number of completions to generate
            temperature: Sampling temperature
            logprobs: Whether to return logprobs (not supported by Fireworks)
            
        Returns:
            The model's response text
        """
        choices = await self._get_response_many(
            messages,
            n=n,
            temperature=temperature,
            logprobs=logprobs
        )
        return choices[0].message.content

    async def _get_response_many(
        self,
        messages: list[Message],
        *,
        n: int = 1,
        temperature: float = 1.0,
        logprobs: bool = False,
    ) -> list[Choice]:
        """Get multiple responses from the model.
        
        Args:
            messages: List of messages to send
            n: Number of completions to generate
            temperature: Sampling temperature
            logprobs: Whether to return logprobs (not supported by Fireworks)
            
        Returns:
            List of model responses
        """
        try:
            completion = await _fireworks_chat_completion(
                model=self.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                n=n,
                temperature=temperature,
                max_tokens=2000,
                top_p=1,
                top_k=50,
            )
            
            return [
                Choice(
                    message=Message(
                        role=choice.message.role,
                        content=choice.message.content or "",
                    ),
                ) for choice in completion.choices
            ]
        except Exception as e:
            raise e

    async def _get_logprobs(self, messages: list[Message]) -> dict[str, float]:
        """Get logprobs for the next token (not supported by Fireworks API).
        
        Args:
            messages: List of messages to send
            
        Returns:
            Empty dict since Fireworks doesn't support logprobs
        """
        raise NotImplementedError("Fireworks API does not support logprobs.")
