from typing import Optional
from pydantic import BaseModel

class RateLimit(BaseModel):
    """Rate limit information from OpenRouter"""
    requests: int
    interval: str

class KeyData(BaseModel):
    """Response data from OpenRouter key information endpoint"""
    label: str
    usage: float
    limit: Optional[float]
    is_free_tier: bool
    rate_limit: RateLimit

class KeyResponse(BaseModel):
    """Full response from OpenRouter key information endpoint"""
    data: KeyData 