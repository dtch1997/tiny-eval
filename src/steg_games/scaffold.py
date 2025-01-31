from abc import ABC, abstractmethod

from steg_games.core.types import Message
from typing import Callable

class ScaffoldingInterface(ABC):
    """ Handles scaffolding to prompt the model and extracting the response"""

    @abstractmethod
    def format_prompt(self, prompt: str) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> str:
        pass


class LambdaScaffold(ScaffoldingInterface):
    """ Directly prompt the model to do things. """

    def __init__(self, format_prompt: Callable[[str], str], parse_response: Callable[[str], str]):
        self._format_prompt = format_prompt
        self._parse_response = parse_response

    def format_prompt(self, prompt: str) -> str:
        return self._format_prompt(prompt)

    def parse_response(self, response: str) -> str:
        return self._parse_response(response)

class DirectScaffold(ScaffoldingInterface):
    """ Directly prompt the model to do things. """

    PROMPT_TEMPLATE = r"""{prompt}

    The last line of your response should be of the following format: 'ANSWER: $ANSWER' (without quotes). Think step by step before answering.
    """.strip()

    def _format_prompt(self, prompt: str) -> str:
        return self.PROMPT_TEMPLATE.format(prompt=prompt)

    def _parse_response(self, response: Message) -> Message:
        return Message(role="assistant", content=response.content.split("ANSWER: ")[1])


class ChainOfThoughtScaffold(ScaffoldingInterface):
    """ Directly prompt the model to do things. """

    PROMPT_TEMPLATE_COT = r"""{prompt}

    The last line of your response should be of the following format: 'ANSWER: $ANSWER' (without quotes). Think step by step before answering.
    """.strip()

    def _format_prompt(self, prompt: str) -> str:
        return self.PROMPT_TEMPLATE_COT.format(prompt=prompt)

    def _parse_response(self, response: Message) -> Message:
        return Message(role="assistant", content=response.content.split("ANSWER: ")[1])