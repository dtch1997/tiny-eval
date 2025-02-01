from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

Prompt = str 
Response = str

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
class Choice:
    message: Message
    logprobs: list[float] | None = None

@dataclass
class MessageHistory:
    messages: list[Message]

Context = list[Message]

@dataclass
class Question:
    context: Context
    prompt: Prompt

    @classmethod
    def from_prompt(cls, prompt: Prompt, context: Context | None = None) -> "Question":
        return cls(context=context or [], prompt=prompt)

    @property 
    def messages(self) -> list[Message]:
        return self.context + [Message(role="user", content=self.prompt)]

@dataclass
class Completion:
    context: Context
    prompt: Prompt
    response: Response
