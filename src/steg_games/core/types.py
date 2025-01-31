from dataclasses import dataclass, asdict
from typing import Literal

from openai.types.chat.chat_completion import Choice as Choice

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(**data)

@dataclass
class Prompt:
    system_message: Message
    context: list[Message] # A list of alternating (user, assistant) messages
    prompt: str

    @property 
    def messages(self) -> list[Message]:
        return [self.system_message] + self.context + [Message(role="user", content=self.prompt)]