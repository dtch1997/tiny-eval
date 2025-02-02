from abc import ABC, abstractmethod
from tiny_eval.core._types import Message, Question

def _get_messages(question: str | Question | list[Message]) -> list[Message]:
    if isinstance(question, str):
        question = Question.from_prompt(question)
    
    if isinstance(question, Question):
        messages = question.messages
    else:
        messages = question
    return messages

class ModelAPIInterface(ABC):
    """
    Abstract class for a model API.
    """

    def get_response(self, question: str | Question | list[Message], **kwargs) -> str:
        messages = _get_messages(question)
        return self._get_response(messages, **kwargs)
    
    def get_logprobs(self, question: str | Question | list[Message]) -> dict[str, float]:
        messages = _get_messages(question)
        return self._get_logprobs(messages)

    @abstractmethod
    async def _get_response(self, messages: list[Message], **kwargs) -> str:
        pass

    @abstractmethod
    async def _get_logprobs(self, messages: list[Message]) -> dict[str, float]:
        pass
