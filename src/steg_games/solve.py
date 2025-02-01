from typing import Awaitable, Callable

from steg_games.core.constants import Model
from steg_games.core.types import Question, Message, MessageHistory
from steg_games.scaffold import ScaffoldingInterface
from steg_games.model_api import ModelAPIInterface

from steg_games.utils.asyncio import chain

def build_solver(
    model: Model,
    model_api: ModelAPIInterface,
    scaffold: ScaffoldingInterface,
) -> Callable[[Question], Awaitable[str]]:
    """ Build a solver using a model and scaffold. """
    
    async def build_message_history(question: Question) -> MessageHistory:
        formatted_prompt = scaffold.format_prompt(question.prompt)
        return MessageHistory(messages=question.context + [Message(role="user", content=formatted_prompt)])

    async def get_response(message_history: MessageHistory) -> str:
        return await model_api.get_response(message_history.messages, model)

    async def get_answer(response: str) -> str:
        return scaffold.parse_response(response)

    return chain(
        build_message_history,
        get_response,
        get_answer,
    )