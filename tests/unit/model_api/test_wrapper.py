import pytest
import asyncio
from unittest.mock import AsyncMock
import time
from collections import deque

from steg_games.model_api.wrapper import RateLimiter
from steg_games.core._types import Message

class MockAPI:
    def __init__(self):
        self.call_times = deque()
        self.get_response = AsyncMock(return_value="test response")
        self.get_logprobs = AsyncMock(return_value={"test": -1.0})

@pytest.fixture
def mock_api():
    return MockAPI()

PERIOD_LENGTH = 0.01

@pytest.fixture
def rate_limiter(mock_api):
    return RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)

@pytest.mark.asyncio
async def test_basic_rate_limiting():
    """Test that requests are rate limited to max_requests_per_minute"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)
    
    start_time = time.time()
    
    # Make 3 requests - the third should be delayed
    await asyncio.gather(
        rate_limiter.get_response([Message(role="user", content="test1")]),
        rate_limiter.get_response([Message(role="user", content="test2")]),
        rate_limiter.get_response([Message(role="user", content="test3")])
    )
    
    elapsed_time = time.time() - start_time
    assert elapsed_time >= PERIOD_LENGTH  # Third request should wait ~60s

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling many concurrent requests"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=10, period_length=PERIOD_LENGTH)
    
    # Create 20 concurrent requests
    requests = [
        rate_limiter.get_response([Message(role="user", content=f"test{i}")])
        for i in range(20)
    ]
    
    start_time = time.time()
    await asyncio.gather(*requests)
    elapsed_time = time.time() - start_time
    
    # Should take at least 1 period to process all requests
    assert elapsed_time >= PERIOD_LENGTH
    assert mock_api.get_response.call_count == 20

@pytest.mark.asyncio
async def test_logprobs_rate_limiting():
    """Test rate limiting applies to logprobs requests"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)
    
    start_time = time.time()
    
    # Make 3 logprob requests
    await asyncio.gather(
        rate_limiter.get_logprobs([Message(role="user", content="test1")]),
        rate_limiter.get_logprobs([Message(role="user", content="test2")]),
        rate_limiter.get_logprobs([Message(role="user", content="test3")])
    )
    
    elapsed_time = time.time() - start_time
    assert elapsed_time >= PERIOD_LENGTH
    assert mock_api.get_logprobs.call_count == 3

@pytest.mark.asyncio
async def test_mixed_request_types():
    """Test rate limiting applies across both response and logprob requests"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)
    
    start_time = time.time()
    
    # Mix response and logprob requests
    await asyncio.gather(
        rate_limiter.get_response([Message(role="user", content="test1")]),
        rate_limiter.get_logprobs([Message(role="user", content="test2")]),
        rate_limiter.get_response([Message(role="user", content="test3")])
    )
    
    elapsed_time = time.time() - start_time
    assert elapsed_time >= PERIOD_LENGTH
    assert mock_api.get_response.call_count == 2
    assert mock_api.get_logprobs.call_count == 1

@pytest.mark.asyncio
async def test_window_sliding():
    """Test that the rate limit window properly slides"""
    mock_api = MockAPI()
    rate_limiter = RateLimiter(mock_api, max_requests_per_period=2, period_length=PERIOD_LENGTH)
    
    # Make 2 requests
    await asyncio.gather(
        rate_limiter.get_response([Message(role="user", content="test1")]),
        rate_limiter.get_response([Message(role="user", content="test2")])
    )
    
    # Wait 2 periods for window to slide
    await asyncio.sleep(2 * PERIOD_LENGTH)
    
    # Should be able to make 2 more requests without waiting
    start_time = time.time()
    await asyncio.gather(
        rate_limiter.get_response([Message(role="user", content="test3")]),
        rate_limiter.get_response([Message(role="user", content="test4")])
    )
    
    elapsed_time = time.time() - start_time
    assert elapsed_time < PERIOD_LENGTH  # Should be nearly instant