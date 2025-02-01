from abc import ABC, abstractmethod
from steg_games.core.types import Message
from steg_games.core.constants import Model

class ModelAPIInterface(ABC):
    """
    Abstract class for a model API.
    """

    @abstractmethod
    async def get_response(self, messages: list[Message], model: Model, **kwargs) -> str:
        pass

    @abstractmethod
    async def get_logprobs(self, messages: list[Message], model: Model) -> dict[str, float]:
        pass