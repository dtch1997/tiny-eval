from enum import Enum
from typing import Literal
from .hashable import HashableBaseModel

MessageRoleType = Literal["system", "assistant", "user"]

class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"

class TokenLogProb(HashableBaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[dict[str, float]]

class LogProbs(HashableBaseModel):
    content: list[TokenLogProb] | None = None
    refusal: list[TokenLogProb] | None = None

class Message(HashableBaseModel):
    role: MessageRole
    content: str
    refusal: str | None = None # returned by OpenAI
