from abc import ABC, abstractmethod
from tiny_eval.core._types import Message

class ModelAPIInterface(ABC):
    """
    Abstract class for a model API.
    """

    @abstractmethod
    async def get_response(self, messages: list[Message], **kwargs) -> str:
        pass

    @abstractmethod
    async def get_logprobs(self, messages: list[Message]) -> dict[str, float]:
        pass
