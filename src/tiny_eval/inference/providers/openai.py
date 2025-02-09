import backoff
import openai

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from ..interface import InferenceAPIInterface
from tiny_eval.inference.types import (
    InferencePrompt,
    InferenceParams,
    InferenceResponse,
    StopReason,
    InferenceChoice,
    Message,
)

def on_backoff(details):
    """We don't print connection error because there's sometimes a lot of them and they're not interesting."""
    exception_details = details["exception"]
    if not str(exception_details).startswith("Connection error."):
        print(exception_details)

@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=on_backoff,
)
async def _openai_chat_completion(
    *,
    client: AsyncOpenAI,
    model: str,
    prompt: InferencePrompt,
    params: InferenceParams,
) -> InferenceResponse:
    """ Lightweight wrapper around OpenAI's chat completion API. """
    message_obj = [message.model_dump() for message in prompt.messages]
    response: ChatCompletion = await client.chat.completions.create(
        model=model,
        messages=message_obj,
        **params.model_dump(),
    )
    
    if response.model != model:
        # This should never happen, but it's good to have a fallback.
        raise RuntimeError(f"OpenAI returned a different model than requested: {response.model} != {model}")

    return InferenceResponse(
        model=response.model,
        choices=[InferenceChoice(
            stop_reason=StopReason(choice.finish_reason),
            message=Message(
                role=choice.message.role,
                content=choice.message.content,
                logprobs=choice.logprobs,
            ),
        ) for choice in response.choices],
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )

class OpenAIInferenceAPI(InferenceAPIInterface):
    """
    Model API for OpenRouter.
    """
    model: str
    client: AsyncOpenAI

    def __init__(
        self,
        model: str,
        client: AsyncOpenAI | None = None,
    ):
        self.model = model
        self.client = client or AsyncOpenAI()
    
    async def _get_response(
        self, 
        prompt: InferencePrompt,
        params: InferenceParams,
    ) -> InferenceResponse:
        return await _openai_chat_completion(
            client=self.client,
            model=self.model,
            prompt=prompt,
            params=params,
        )