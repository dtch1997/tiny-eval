import pytest
from typing import List

from tiny_eval.core.constants import Model
from tiny_eval.inference import get_response
from tiny_eval.inference.types import InferencePrompt
from tiny_eval.core.messages import Message, MessageRole

@pytest.fixture
def test_prompt() -> str:
    """Simple test prompt for model queries."""
    return "What is 2+2?"

@pytest.fixture
def test_prompt_messages() -> InferencePrompt:
    """Test prompt using message format."""
    return InferencePrompt(messages=[
        Message(role=MessageRole.user, content="What is 2+2?")
    ])

@pytest.mark.asyncio
async def test_openai_model_response(test_prompt: str) -> None:
    """Test that OpenAI models can be queried successfully."""
    model = Model.GPT_4o_mini
    response = await get_response(model, test_prompt)
    
    # Basic response validation
    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response.lower()

@pytest.mark.asyncio
async def test_openrouter_model_response(test_prompt: str) -> None:
    """Test that OpenRouter models can be queried successfully."""
    model = Model.CLAUDE_3_5_SONNET
    response = await get_response(model, test_prompt)
    
    # Basic response validation
    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response.lower()

@pytest.mark.asyncio
async def test_message_format_prompt(test_prompt_messages: InferencePrompt) -> None:
    """Test that models accept InferencePrompt format."""
    model = Model.GPT_4o_mini
    response = await get_response(model, test_prompt_messages)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response.lower()

@pytest.mark.parametrize("model", [
    Model.GPT_4o_mini,
    Model.CLAUDE_3_5_SONNET,
    Model.DEEPSEEK_CHAT,
    Model.QWEN_2_5_72B_INSTRUCT,
    pytest.param(Model.GROK_2, marks=pytest.mark.xfail(reason="Grok API may be unstable")),
])
@pytest.mark.asyncio
async def test_all_models(model: Model, test_prompt: str) -> None:
    """Test that all supported models can be queried.
    
    Note: Some models may be marked as expected to fail if they're unstable or in beta.
    """
    response = await get_response(model, test_prompt)
    
    assert isinstance(response, str)
    assert len(response) > 0
    # The response should contain "4" since it's a simple math question
    assert "4" in response.lower()

@pytest.mark.asyncio
async def test_model_string_input(test_prompt: str) -> None:
    """Test that models can be specified as strings."""
    model = "openai/gpt-4o-mini-2024-07-18"  # String version of Model.GPT_4o_mini
    response = await get_response(model, test_prompt)
    
    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response.lower()

@pytest.mark.asyncio
async def test_error_handling_invalid_model() -> None:
    """Test that invalid model names are handled appropriately."""
    with pytest.raises(ValueError):
        await get_response("invalid-model", "test prompt") 