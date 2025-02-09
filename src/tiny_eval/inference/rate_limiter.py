import time
import asyncio
from collections import deque

from tiny_eval.inference.interface import InferenceAPIInterface

class RateLimiter(InferenceAPIInterface):
    """Rate limits requests to the underlying model API."""
    
    def __init__(self, api: InferenceAPIInterface, max_requests_per_period: int, period_length: float):
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

    async def _get_response(self, *args, **kwargs) -> str:
        await self._wait_if_needed()
        return await self.api._get_response(*args, **kwargs)
