import pytest
from unittest.mock import AsyncMock, patch
import json
from pathlib import Path
import tempfile
import shutil

from tiny_eval.inference.cache import InferenceCache
from tiny_eval.inference.types import InferencePrompt, InferenceParams, InferenceResponse, StopReason
from tiny_eval.inference.interface import InferenceAPIInterface
from tiny_eval.core.messages import Message, MessageRole

@pytest.fixture
def mock_api():
    # Create a mock that can be called directly
    api = AsyncMock(spec=InferenceAPIInterface)
    # Configure the mock's __call__ behavior
    api.return_value = None  # Will be overridden in tests
    return api

@pytest.fixture
def cache(mock_api):
    return InferenceCache(api=mock_api)

@pytest.fixture
def sample_prompt():
    return InferencePrompt(messages=[Message(role=MessageRole.user, content="test prompt")])

@pytest.fixture
def sample_params():
    return InferenceParams(
        temperature=0.7,
        max_completion_tokens=100
    )

@pytest.fixture
def sample_response():
    return InferenceResponse(
        model="test-model",
        choices=[{
            "message": {
                "role": "assistant",
                "content": "test response"
            },
            "stop_reason": StopReason.STOP_SEQUENCE
        }],
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15
    )

@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.mark.asyncio
async def test_cache_miss(cache, mock_api, sample_prompt, sample_params, sample_response):
    """Test that cache miss calls the underlying API"""
    mock_api.return_value = sample_response
    
    result = await cache(sample_prompt, sample_params)
    
    assert result == sample_response
    mock_api.assert_called_once_with(sample_prompt, sample_params)

@pytest.mark.asyncio
async def test_cache_hit(cache, mock_api, sample_prompt, sample_params, sample_response):
    """Test that cache hit returns cached response without calling API"""
    mock_api.return_value = sample_response
    
    # First call to populate cache
    await cache(sample_prompt, sample_params)
    mock_api.reset_mock()
    
    # Second call should use cache
    result = await cache(sample_prompt, sample_params)
    
    assert result == sample_response
    mock_api.assert_not_called()

@pytest.mark.asyncio
async def test_different_params_different_cache(
    cache, mock_api, sample_prompt, sample_params, sample_response
):
    """Test that different params result in different cache entries"""
    mock_api.return_value = sample_response
    different_params = InferenceParams(temperature=0.8, max_completion_tokens=100)
    
    # Call with first params
    await cache(sample_prompt, sample_params)
    mock_api.reset_mock()
    
    # Call with different params should miss cache
    await cache(sample_prompt, different_params)
    
    mock_api.assert_called_once_with(sample_prompt, different_params)

@pytest.mark.asyncio
async def test_different_prompts_different_cache(
    cache, mock_api, sample_prompt, sample_params, sample_response
):
    """Test that different prompts result in different cache entries"""
    mock_api.return_value = sample_response
    different_prompt = InferencePrompt(messages=[Message(role=MessageRole.user, content="different prompt")])
    
    # Call with first prompt
    await cache(sample_prompt, sample_params)
    mock_api.reset_mock()
    
    # Call with different prompt should miss cache
    await cache(different_prompt, sample_params)
    
    mock_api.assert_called_once_with(different_prompt, sample_params)

@pytest.mark.asyncio
async def test_cache_persistence(
    mock_api, sample_prompt, sample_params, sample_response, temp_cache_dir
):
    """Test that cache is saved to and loaded from disk."""
    # Create first cache instance and make a call
    mock_api.return_value = sample_response
    cache1 = InferenceCache(api=mock_api, cache_dir=temp_cache_dir)
    await cache1(sample_prompt, sample_params)
    
    # Force save
    cache1.save_cache()
    
    # Verify cache file exists and contains expected data
    assert cache1.cache_path.exists()
    cache_data = json.loads(cache1.cache_path.read_text())
    assert len(cache_data) == 1
    
    # Create new cache instance - should load from disk
    cache2 = InferenceCache(api=mock_api, cache_dir=temp_cache_dir)
    mock_api.reset_mock()
    
    # Make same request - should use cached value
    result = await cache2(sample_prompt, sample_params)
    assert result == sample_response
    mock_api.assert_not_called()

@pytest.mark.asyncio
async def test_invalid_cache_file(mock_api, temp_cache_dir):
    """Test that invalid cache file is handled gracefully."""
    cache_path = temp_cache_dir / "inference_cache.json"
    
    # Write invalid JSON
    cache_path.write_text("invalid json")
    
    # Should not raise error, should start with empty cache
    cache = InferenceCache(api=mock_api, cache_dir=temp_cache_dir)
    assert len(cache._cache) == 0

def test_default_cache_path(mock_api):
    """Test that default cache path is set correctly."""
    # Override CACHE_DIR for this test
    with patch('tiny_eval.inference.cache.CACHE_DIR', Path.home() / ".cache" / "tiny_eval"):
        cache = InferenceCache(api=mock_api)
        expected_path = Path.home() / ".cache" / "tiny_eval" / "inference_cache.json"
        assert cache.cache_path == expected_path
