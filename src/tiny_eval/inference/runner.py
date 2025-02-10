from openai import AsyncOpenAI
from tiny_eval.core.constants import Model
from tiny_eval.inference.interface import InferenceAPIInterface
from tiny_eval.inference.openai.api import OpenAIInferenceAPI
from tiny_eval.inference.openrouter.api import OpenRouterInferenceAPI
from tiny_eval.inference.data_models import InferenceParams, InferencePrompt
from tiny_eval.core.constants import OPENAI_API_KEY
from tiny_eval.core.messages import Message, MessageRole
from typing import Union


def build_inference_api(model: Union[str, Model]) -> InferenceAPIInterface:
    """
    Build an inference API for a given model
    
    Args:
        model: Model identifier, either as string or Model enum
        
    Returns:
        InferenceAPIInterface: Appropriate API client for the model
    """
    if isinstance(model, str):
        model = Model(model)

    if model.value.startswith("openai/"):
        return OpenAIInferenceAPI()
    else:
        # Use OpenRouter for all non-OpenAI models
        return OpenRouterInferenceAPI()


async def get_response(
    model: Union[str, Model],
    prompt: str | InferencePrompt,
    params: InferenceParams | None = None,
) -> str:
    """
    Get a response from the inference API.
    
    Args:
        model: The model identifier to use (e.g., "gpt-4", "claude-3")
        prompt: The prompt string or InferencePrompt object to send
        params: Optional inference parameters
        
    Returns:
        str: The response content as a string
    """
    if isinstance(model, str):
        model = Model(model)
    api = build_inference_api(model)
    if isinstance(prompt, str):
        prompt = InferencePrompt(messages=[Message(role=MessageRole.user, content=prompt)])
    params = params or InferenceParams()
    response = await api(model.value, prompt, params)
    return response.choices[0].message.content