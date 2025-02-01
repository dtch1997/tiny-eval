from typing import Awaitable, Callable

from steg_games.core.constants import Model
from steg_games.core.types import Question, Message, Choice, MessageHistory
from steg_games.scaffold import ScaffoldingInterface
from steg_games.model_api import ModelAPIInterface

from steg_games.utils.asyncio import batch_chain

def build_solver(
    model: Model,
    model_api: ModelAPIInterface,
    scaffold: ScaffoldingInterface,
) -> Callable[[Question], Awaitable[list[str]]]:
    """ Build a solver using a model and scaffold. """
    
    async def build_messages(question: Question) -> MessageHistory:
        formatted_prompt = scaffold.format_prompt(question.prompt)
        return MessageHistory(messages=question.context + [Message(role="user", content=formatted_prompt)])

    async def get_response(message_history: MessageHistory) -> list[str]:
        choices: list[Choice] = await model_api.get_response(message_history.messages, model)
        return [choice.message.content for choice in choices]

    async def get_answer(response: str) -> str:
        return scaffold.parse_response(response)

    return batch_chain(
        build_messages,
        get_response,
        get_answer,
    )