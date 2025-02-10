import time
import asyncio
from typing import TypeVar, Callable, ParamSpec, Awaitable
from functools import wraps

P = ParamSpec('P')
T = TypeVar('T')

class AsyncRateLimiter:
    """
    Rate limiter for async functions using token bucket algorithm
    """
    def __init__(self, requests: int, window: int):
        """
        Initialize rate limiter
        
        Args:
            requests: Maximum number of requests allowed
            window: Time window in seconds
        """
        self.max_tokens = requests
        self.window = window
        self._tokens = requests
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def _update_tokens(self) -> None:
        """Updates the token count based on elapsed time"""
        now = time.monotonic()
        time_passed = now - self._last_update
        self._tokens = min(
            self.max_tokens,
            self._tokens + int((time_passed * self.max_tokens) / self.window)
        )
        self._last_update = now

    async def _acquire_token(self) -> None:
        """Acquires a token for rate limiting, waiting if necessary"""
        async with self._lock:
            await self._update_tokens()
            while self._tokens <= 0:
                wait_time = 0.1  # 100ms
                await asyncio.sleep(wait_time)
                await self._update_tokens()
            self._tokens -= 1
    
    def __call__(
        self,
        func: Callable[P, Awaitable[T]]
    ) -> Callable[P, Awaitable[T]]:
        """
        Decorator that applies rate limiting to an async function
        
        Args:
            func: Async function to rate limit
            
        Returns:
            Rate limited async function
        """
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            await self._acquire_token()
            return await func(*args, **kwargs)
        return wrapper 