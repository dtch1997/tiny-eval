import backoff
import openai
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from typing import cast

from ..interface import InferenceAPIInterface
from tiny_eval.inference.types import (
    InferencePrompt,
    InferenceParams,
    InferenceResponse,
    StopReason,
    InferenceChoice,
    Message,
)
from tiny_eval.core.messages import MessageRole

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
    """Lightweight wrapper around OpenAI's chat completion API."""
    # Convert messages to OpenAI format
    messages: list[ChatCompletionMessageParam] = []
    for message in prompt.messages:
        msg_dict = {
            "role": message.role.value,
            "content": message.content
        }
        if message.role == MessageRole.system:
            messages.append(cast(ChatCompletionSystemMessageParam, msg_dict))
        elif message.role == MessageRole.user:
            messages.append(cast(ChatCompletionUserMessageParam, msg_dict))
        elif message.role == MessageRole.assistant:
            messages.append(cast(ChatCompletionAssistantMessageParam, msg_dict))

    # Filter out logprobs from params if present
    params_dict = params.model_dump(exclude_none=True)
    params_dict.pop('logprobs', None)
    params_dict.pop('top_logprobs', None)

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **params_dict,
    )

    # Ensure usage is not None before accessing
    usage = response.usage
    if usage is None:
        raise ValueError("Response usage information is missing")

    return InferenceResponse(
        model=response.model,
        choices=[InferenceChoice(
            stop_reason=StopReason(choice.finish_reason),
            message=Message(
                role=choice.message.role,
                content=choice.message.content,
            ),
        ) for choice in response.choices],
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )

class OpenAIInferenceAPI(InferenceAPIInterface):
    """
    Inference API for OpenAI with rate limiting
    """
    # TODO: support multiple clients
    client: AsyncOpenAI

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
    ):
        self.client = client or AsyncOpenAI()
    
    async def __call__(
        self, 
        model: str,
        prompt: InferencePrompt,
        params: InferenceParams,
    ) -> InferenceResponse:
        
        if model.startswith("openai/"):
            model = model.replace("openai/", "")

        return await _openai_chat_completion(
            client=self.client,
            model=model,
            prompt=prompt,
            params=params,
        )