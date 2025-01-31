from dataclasses import dataclass, asdict
from typing import Literal

from openai.types.chat.chat_completion import Choice as Choice
from collections import namedtuple

Prompt = str 
Response = str
Completion = namedtuple("Completion", ["prompt", "response"])

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(**data)