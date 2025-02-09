import pytest
import asyncio
from unittest.mock import AsyncMock
import time
from collections import deque

from tiny_eval.inference.rate_limiter import RateLimiter
from tiny_eval.inference.types import InferencePrompt, InferenceParams, InferenceResponse, InferenceChoice, StopReason
from tiny_eval.core.messages import Message, MessageRole

class MockAPI:
    def __init__(self):
        self.call_times = deque()
        self._mock = AsyncMock(return_value=InferenceResponse(
            model="test-model",
            choices=[InferenceChoice(
                stop_reason=StopReason.STOP_SEQUENCE,
                message=Message(role=MessageRole.assistant, content="test response")
            )],
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20
        ))

    async def __call__(self, *args, **kwargs):
        return await self._mock(*args, **kwargs)

    @property
    def call_count(self):
        return self._mock.call_count

    def reset_mock(self):
        self._mock.reset_mock()

@pytest.fixture
def mock_api():
    return MockAPI()

PERIOD_LENGTH = 0.01

@pytest.fixture
def rate_limiter(mock_api):
    return RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)

def make_test_prompt(content: str) -> InferencePrompt:
    """Helper to create test prompts"""
    return InferencePrompt(messages=[Message(role=MessageRole.user, content=content)])

def make_test_params() -> InferenceParams:
    """Helper to create test params"""
    return InferenceParams()

@pytest.mark.asyncio
async def test_basic_rate_limiting():
    """Test that requests are rate limited to max_requests_per_period"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)
    
    start_time = time.time()
    
    # Make 3 requests - the third should be delayed
    await asyncio.gather(
        rate_limiter(make_test_prompt("test1"), make_test_params()),
        rate_limiter(make_test_prompt("test2"), make_test_params()),
        rate_limiter(make_test_prompt("test3"), make_test_params())
    )
    
    elapsed_time = time.time() - start_time
    assert elapsed_time >= PERIOD_LENGTH  # Third request should wait for next period

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling many concurrent requests"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=10, period_length=PERIOD_LENGTH)
    
    # Create 20 concurrent requests
    requests = [
        rate_limiter(make_test_prompt(f"test{i}"), make_test_params())
        for i in range(20)
    ]
    
    start_time = time.time()
    await asyncio.gather(*requests)
    elapsed_time = time.time() - start_time
    
    # Should take at least 1 period to process all requests
    assert elapsed_time >= PERIOD_LENGTH
    assert mock_api.call_count == 20

@pytest.mark.asyncio
async def test_window_sliding():
    """Test that the rate limit window properly slides"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)
    
    # Make 2 requests
    await asyncio.gather(
        rate_limiter(make_test_prompt("test1"), make_test_params()),
        rate_limiter(make_test_prompt("test2"), make_test_params())
    )
    
    # Wait 2 periods for window to slide
    await asyncio.sleep(2 * PERIOD_LENGTH)
    
    # Should be able to make 2 more requests without waiting
    start_time = time.time()
    await asyncio.gather(
        rate_limiter(make_test_prompt("test3"), make_test_params()),
        rate_limiter(make_test_prompt("test4"), make_test_params())
    )
    
    elapsed_time = time.time() - start_time
    assert elapsed_time < PERIOD_LENGTH  # Should be nearly instant