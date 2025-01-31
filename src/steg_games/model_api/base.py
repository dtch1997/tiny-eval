from abc import ABC, abstractmethod
from steg_games.core.types import Message, Choice


class ModelAPIInterface(ABC):
    """
    Abstract class for a model API.
    """

    @abstractmethod
    async def get_response(self, messages: list[Message], model: str) -> list[Choice]:
        pass
