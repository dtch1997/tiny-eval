import requests
from tiny_eval.core.constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from tiny_eval.inference.openrouter.data_models import KeyResponse

def check_openrouter_rate_limit() -> KeyResponse:
    """
    Fetches rate limit and usage information from OpenRouter API.
    
    Returns:
        KeyResponse: Contains information about API key usage, limits, and rate limiting
        
    Raises:
        requests.RequestException: If the request fails
        ValueError: If OPENROUTER_API_KEY is not set
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY must be set")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    
    response = requests.get(
        f"{OPENROUTER_BASE_URL}/auth/key",
        headers=headers
    )
    response.raise_for_status()
    data = response.json()
    return KeyResponse.model_validate(data)

if __name__ == "__main__":
    print(check_openrouter_rate_limit())