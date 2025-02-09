from abc import ABC, abstractmethod

from tiny_eval.inference.types import InferencePrompt, InferenceParams, InferenceResponse

class InferenceAPIInterface(ABC):
    """
    Abstract class for an inference API.
    """

    @abstractmethod
    async def _get_response(self, prompt: InferencePrompt, params: InferenceParams) -> InferenceResponse:
        pass