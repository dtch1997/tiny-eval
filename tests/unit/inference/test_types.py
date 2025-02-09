"""
Unit tests for types in src/tiny_eval/inference/types.py

These tests verify that the objects can be instantiated and that the hash methods
(from HashableBaseModel) return deterministic results.
"""

import pytest

from tiny_eval.inference.types import (
    InferenceParams,
    InferencePrompt,
    InferenceChoice,
    InferenceResponse,
    StopReason,
)
from tiny_eval.core.messages import Message, MessageRole


def test_inference_params_hash() -> None:
    """
    Test that InferenceParams can be initialized and hashed deterministically.

    Checks that:
    - Two identical instances produce the same SHA1 hash.
    - Changing a field produces a different hash.
    """
    params1 = InferenceParams(temperature=0.7, top_p=0.9, max_completion_tokens=150)
    params2 = InferenceParams(temperature=0.7, top_p=0.9, max_completion_tokens=150)

    hash1 = params1.model_hash()
    hash2 = params2.model_hash()

    # Validate that a hash is produced, and that it is a 40-character SHA1 hex digest.
    assert isinstance(hash1, str)
    assert len(hash1) == 40
    # Hashes of identical models should be the same.
    assert hash1 == hash2

    # Changing a field should yield a different hash.
    params3 = InferenceParams(temperature=0.8, top_p=0.9, max_completion_tokens=150)
    hash3 = params3.model_hash()
    assert hash1 != hash3


def test_inference_prompt_hash() -> None:
    """
    Test that InferencePrompt can be initialized and hashed deterministically.

    Verifies that:
    - Two instances with the same messages generate the same hash.
    - Changing the messages produces a different hash.
    """
    msg1 = Message(role=MessageRole.user, content="Hello world!")
    msg2 = Message(role=MessageRole.assistant, content="Hi there!")
    prompt1 = InferencePrompt(messages=[msg1, msg2])
    prompt2 = InferencePrompt(messages=[msg1, msg2])

    hash1 = prompt1.model_hash()
    hash2 = prompt2.model_hash()

    assert isinstance(hash1, str)
    assert len(hash1) == 40
    assert hash1 == hash2

    # Change one of the messages and verify that the hash differs.
    msg3 = Message(role=MessageRole.user, content="Different")
    prompt3 = InferencePrompt(messages=[msg3, msg2])
    hash3 = prompt3.model_hash()

    assert hash1 != hash3


def test_inference_choice_initialization() -> None:
    """
    Test that InferenceChoice objects can be initialized correctly and that
    the stop_reason field is parsed as expected.
    """
    msg = Message(role=MessageRole.assistant, content="Response content")
    # Using a string for stop_reason should be parsed to STOP_SEQUENCE.
    choice = InferenceChoice(stop_reason="stop", message=msg)
    assert choice.stop_reason == StopReason.STOP_SEQUENCE

    # Using an enum value directly.
    choice2 = InferenceChoice(stop_reason=StopReason.API_ERROR, message=msg)
    assert choice2.stop_reason == StopReason.API_ERROR


def test_inference_response_initialization() -> None:
    """
    Test that InferenceResponse objects can be initialized properly.
    """
    msg = Message(role=MessageRole.assistant, content="Response content")
    choice = InferenceChoice(stop_reason="stop", message=msg)
    response = InferenceResponse(
        model="test-model",
        choices=[choice],
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    assert response.model == "test-model"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.total_tokens == 15

    # Verify that the 'choices' list contains our choice with the correct stop_reason.
    assert response.choices[0].stop_reason == StopReason.STOP_SEQUENCE


def test_inference_params_model_dump() -> None:
    """Test that model_dump works correctly for InferenceParams."""
    params = InferenceParams(temperature=0.7, top_p=0.9, max_completion_tokens=150)
    params_dict = params.model_dump()
    
    # Only check the fields we explicitly set, since there are many defaults
    assert params_dict["temperature"] == 0.7
    assert params_dict["top_p"] == 0.9
    assert params_dict["max_completion_tokens"] == 150


def test_inference_prompt_model_dump() -> None:
    """Test that model_dump works correctly for InferencePrompt."""
    msg = Message(role=MessageRole.user, content="Hello")
    prompt = InferencePrompt(messages=[msg])
    prompt_dict = prompt.model_dump()
    
    # Check the structure matches but allow for None fields
    assert len(prompt_dict["messages"]) == 1
    message = prompt_dict["messages"][0]
    assert message["content"] == "Hello"
    assert message["role"] == MessageRole.user
    assert "refusal" in message


def test_inference_choice_model_dump() -> None:
    """Test that model_dump works correctly for InferenceChoice."""
    choice_msg = Message(role=MessageRole.assistant, content="Hi")
    choice = InferenceChoice(stop_reason=StopReason.STOP_SEQUENCE, message=choice_msg)
    
    choice_dict = choice.model_dump()
    
    assert choice_dict["message"]["content"] == "Hi"
    assert choice_dict["message"]["role"] == MessageRole.assistant
    assert choice_dict["stop_reason"] == StopReason.STOP_SEQUENCE


def test_inference_response_model_dump() -> None:
    """Test that model_dump works correctly for InferenceResponse."""
    choice_msg = Message(role=MessageRole.assistant, content="Hi")
    choice = InferenceChoice(stop_reason=StopReason.STOP_SEQUENCE, message=choice_msg)
    response = InferenceResponse(
        model="test-model",
        choices=[choice],
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    response_dict = response.model_dump()
    
    # Check core fields
    assert response_dict["model"] == "test-model"
    assert response_dict["prompt_tokens"] == 10
    assert response_dict["completion_tokens"] == 5
    assert response_dict["total_tokens"] == 15
    
    # Check choices structure
    assert len(response_dict["choices"]) == 1
    choice = response_dict["choices"][0]
    assert choice["message"]["content"] == "Hi"
    assert choice["message"]["role"] == MessageRole.assistant
    assert choice["stop_reason"] == StopReason.STOP_SEQUENCE 