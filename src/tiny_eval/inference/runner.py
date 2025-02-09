from openai import AsyncOpenAI
from tiny_eval.core.constants import Model
from tiny_eval.inference.interface import InferenceAPIInterface
from tiny_eval.inference.openai.api import OpenAIInferenceAPI
from tiny_eval.inference.types import InferenceParams, InferencePrompt
from tiny_eval.core.constants import OPENAI_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from tiny_eval.core.messages import Message, MessageRole


def build_inference_api(model: Model) -> InferenceAPIInterface:
    """Build an inference API for a given model"""

    if model.value.startswith("openai/"):                  
        return OpenAIInferenceAPI(
            client=AsyncOpenAI(api_key=OPENAI_API_KEY),
        )
    else:
        return OpenAIInferenceAPI(
            client=AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            ),
        )


async def get_response(
    model: str,
    prompt: str | InferencePrompt,
    params: InferenceParams | None = None,
) -> str:
    """
    Get a response from the inference API.
    
    Args:
        api: The inference API to use
        model: The model identifier to use (e.g., "gpt-4", "claude-3")
        prompt: The prompt string to send
        params: Optional inference parameters
        
    Returns:
        The response content as a string
    """
    if isinstance(prompt, str):
        prompt = InferencePrompt(messages=[Message(role=MessageRole.user, content=prompt)])
    params = params or InferenceParams()
    api = build_inference_api(model)
    response = await api(model, prompt, params)
    return response.choices[0].message.content