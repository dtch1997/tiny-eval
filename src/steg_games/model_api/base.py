from abc import ABC, abstractmethod

class ModelAPIInterface(ABC):
    """
    Abstract class for a model API.
    """

    @abstractmethod
    async def get_response(self, prompt: str, model: str) -> str:
        pass
