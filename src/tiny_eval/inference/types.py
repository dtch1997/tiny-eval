import pydantic

from enum import Enum
from typing import Any, Literal, Sequence
from typing_extensions import Self

from tiny_eval.core.messages import Message, LogProbs
from tiny_eval.core.hashable import HashableBaseModel

class InferenceParams(HashableBaseModel):
    """Parameters for LLM API calls, matching OpenAI's chat completion API.
    
    All fields are optional unless marked as Required.
    """
    # Optional parameters with defaults
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    frequency_penalty: float | None = 0.0
    presence_penalty: float | None = 0.0
    stream: bool = False
    
    # Optional parameters without defaults
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    user: str | None = None
    
    # Response format and tools
    response_format: Any = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool = True
    
    # Additional parameters for specific models/features
    store: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = "medium"
    metadata: dict[str, str] | None = pydantic.Field(
        default=None,
        json_schema_extra={
            "max_length_keys": 64,
            "max_length_values": 512,
            "max_items": 16
        }
    )
    modalities: list[str] | None = None
    service_tier: Literal["auto", "default"] | None = "auto"
    stream_options: dict | None = None

    # configuration
    model_config = pydantic.ConfigDict(
        extra="forbid",
        validate_default=True
    )
    
    @pydantic.field_validator("frequency_penalty", "presence_penalty")
    def validate_penalty(cls, v: float | None) -> float | None:
        """Validate that penalty values are between -2.0 and 2.0"""
        if v is not None and not -2.0 <= v <= 2.0:
            raise ValueError("Penalty values must be between -2.0 and 2.0")
        return v
    
    @pydantic.field_validator("top_logprobs")
    def validate_top_logprobs(cls, v: int | None) -> int | None:
        """Validate that top_logprobs is between 0 and 20"""
        if v is not None and not 0 <= v <= 20:
            raise ValueError("top_logprobs must be between 0 and 20")
        return v
    
    @pydantic.field_validator("metadata")
    def validate_metadata(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        """Validate metadata key-value lengths"""
        if v is not None:
            for key, value in v.items():
                if len(key) > 64:
                    raise ValueError("Metadata keys must be <= 64 characters")
                if len(value) > 512:
                    raise ValueError("Metadata values must be <= 512 characters")
            if len(v) > 16:
                raise ValueError("Maximum 16 metadata key-value pairs allowed")
        return v


class StopReason(Enum):
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value

class InferenceChoice(pydantic.BaseModel):
    stop_reason: StopReason
    message: Message
    logprobs: LogProbs | None = None

    @pydantic.field_validator("stop_reason", mode="before")
    def parse_stop_reason(cls, v: str | StopReason) -> StopReason:
        # If already a StopReason, return it as-is.
        if isinstance(v, StopReason):
            return v

        if v in ["length"]:
            return StopReason.MAX_TOKENS
        elif v in ["stop"]:
            return StopReason.STOP_SEQUENCE
        elif v in ["content_filter"]:
            return StopReason.CONTENT_FILTER
        elif v in ["tool_call", "function_call"]:
            return StopReason.TOOL_CALL
        elif v in ["api_error"]:
            return StopReason.API_ERROR
        raise ValueError(f"Invalid stop reason: {v}")


class InferenceResponse(pydantic.BaseModel):
    # metadata
    model: str
    choices: list[InferenceChoice]

    # usage statistics
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    model_config = pydantic.ConfigDict(protected_namespaces=())

class InferencePrompt(HashableBaseModel):
    messages: list[Message]

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            out += f"\n\n{msg.role.value}: {msg.content}"
        return out.strip()

    def __add__(self, other: Self) -> Self:
        return self.__class__(messages=list(self.messages) + list(other.messages))