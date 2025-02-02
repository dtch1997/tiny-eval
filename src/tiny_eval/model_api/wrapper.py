import time
import asyncio
from collections import deque

from tiny_eval.model_api import ModelAPIInterface
from tiny_eval.core._types import Message

class RateLimiter(ModelAPIInterface):
    """Rate limits requests to the underlying model API."""
    
    def __init__(self, api: ModelAPIInterface, max_requests_per_period: int, period_length: float):
        """
        Initialize the rate limiter.
        
        Args:
            api: The underlying ModelAPI implementation
            max_requests_per_period: Maximum number of requests allowed per period
            period_length: Length of the rate limiting period (in seconds)
        """
        self.api = api
        self.max_requests_per_period = max_requests_per_period
        self.period_length = period_length
        self.request_times = deque()

    async def _wait_if_needed(self):
        """
        Wait if necessary to maintain the rate limit.
        Removes old timestamps and checks if we need to wait before next request.
        """
        current_time = time.time()
        
        # Remove timestamps older than our window
        while self.request_times and current_time - self.request_times[0] >= self.period_length:
            self.request_times.popleft()
        
        # If we've hit our limit, wait until we can make another request
        if len(self.request_times) >= self.max_requests_per_period:
            wait_time = self.request_times[0] + self.period_length - current_time
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Add current request timestamp
        self.request_times.append(current_time)

    async def get_response(self, messages: list[Message], **kwargs) -> str:
        """
        Get a response from the API while respecting rate limits.
        
        Args:
            messages: List of messages to send to the API
            **kwargs: Additional arguments to pass to the underlying API
        
        Returns:
            The API response
        """
        await self._wait_if_needed()
        return await self.api.get_response(messages, **kwargs)

    async def get_logprobs(self, messages: list[Message]) -> dict[str, float]:
        """
        Get logprobs from the API while respecting rate limits.
        
        Args:
            messages: List of messages to get logprobs for
        
        Returns:
            Dictionary of token to logprob mappings
        """
        await self._wait_if_needed()
        return await self.api.get_logprobs(messages)