import pytest
from openai import AsyncOpenAI
from typing import AsyncGenerator

from tiny_eval.inference.providers.openai import OpenAIInferenceAPI
from tiny_eval.inference.types import InferencePrompt, InferenceParams
from tiny_eval.core.messages import Message, MessageRole

@pytest.fixture
def openai_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client for testing."""
    return AsyncOpenAI()

@pytest.fixture
def api(openai_client: AsyncOpenAI) -> OpenAIInferenceAPI:
    """Create an OpenAIInferenceAPI instance for testing."""
    return OpenAIInferenceAPI(
        model="gpt-3.5-turbo",
        client=openai_client
    )

@pytest.fixture
def sample_prompt() -> InferencePrompt:
    """Create a sample prompt for testing."""
    return InferencePrompt(messages=[
        Message(role=MessageRole.user, content="What is 2+2?")
    ])

@pytest.fixture
def sample_params() -> InferenceParams:
    """Create sample parameters for testing."""
    return InferenceParams(
        temperature=0.7,
        max_completion_tokens=100
    )

@pytest.mark.asyncio
async def test_openai_api_initialization(openai_client: AsyncOpenAI):
    """Test that OpenAIInferenceAPI can be initialized correctly."""
    api = OpenAIInferenceAPI(
        model="gpt-3.5-turbo",
        client=openai_client
    )
    assert api.model == "gpt-3.5-turbo"
    assert api.client == openai_client

@pytest.mark.asyncio
async def test_openai_api_response(
    api: OpenAIInferenceAPI,
    sample_prompt: InferencePrompt,
    sample_params: InferenceParams
):
    """Test that OpenAIInferenceAPI can make successful API calls."""
    response = await api._get_response(sample_prompt, sample_params)
    
    # Verify response structure
    assert response.model.startswith("gpt-3.5-turbo")
    assert len(response.choices) > 0
    assert response.choices[0].message.role == MessageRole.assistant
    assert response.choices[0].message.content is not None
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens

@pytest.mark.asyncio
async def test_openai_api_error_handling(
    api: OpenAIInferenceAPI,
    sample_prompt: InferencePrompt
):
    """Test that OpenAIInferenceAPI handles invalid parameters appropriately."""
    invalid_params = InferenceParams(temperature=2.5)  # Temperature > 2.0 is invalid
    
    with pytest.raises(Exception):
        await api._get_response(sample_prompt, invalid_params) 