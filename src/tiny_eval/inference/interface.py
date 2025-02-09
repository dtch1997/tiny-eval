from abc import ABC, abstractmethod

from tiny_eval.core.messages import Message, MessageRole
from tiny_eval.inference.types import InferencePrompt, InferenceParams, InferenceResponse

class InferenceAPIInterface(ABC):
    """
    Abstract class for an inference API.
    """

    @abstractmethod
    async def __call__(self, prompt: InferencePrompt, params: InferenceParams) -> InferenceResponse:
        pass


async def get_response(api: InferenceAPIInterface, prompt: str | InferencePrompt, params: InferenceParams | None = None) -> str:
    """
    Get a response from the inference API.
    
    Args:
        api: The inference API to use
        prompt: The prompt string to send
        params: Optional inference parameters
        
    Returns:
        The response content as a string
    """
    if isinstance(prompt, str):
        prompt = InferencePrompt(messages=[Message(role=MessageRole.user, content=prompt)])
    params = params or InferenceParams()
    response = await api(prompt, params)
    return response.choices[0].message.content