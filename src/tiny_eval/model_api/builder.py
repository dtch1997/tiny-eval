from openai import AsyncOpenAI
from tiny_eval.core.constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, OPENAI_API_KEY
from tiny_eval.model_api.openai import OpenAIModelAPI
from tiny_eval.model_api.wrapper import RateLimiter
from tiny_eval.model_api.base import ModelAPIInterface
from tiny_eval.core.constants import Model

# Global dict to store clients
_model_apis: dict[str, ModelAPIInterface] = {}

def build_model_api(model: Model) -> ModelAPIInterface:
    if model.value.startswith("openai/"):
        openai_model_name = model.value.split("/")[1]
        
        # Get or create OpenAI client
        if openai_model_name in _model_apis:
            return _model_apis[openai_model_name]
                    
        model_api = OpenAIModelAPI(openai_model_name, client=AsyncOpenAI(api_key=OPENAI_API_KEY))
        # OpenAI rate limit with Tier 3 API key: 5000 requests per minute
        # Reference: https://platform.openai.com/docs/guides/rate-limits?tier=tier-three
        model_api = RateLimiter(model_api, max_requests_per_period=5000, period_length=60)
        _model_apis[openai_model_name] = model_api
        return model_api

    else:
        model_name = model.value

        # Get or create OpenRouter client
        if model_name in _model_apis:
            return _model_apis[model_name]

        model_api = OpenAIModelAPI(
            model_name,
            client=AsyncOpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=OPENROUTER_API_KEY,
                ),
        )
        # TODO: Check rate limit for OpenRouter, implement rate limiter
        _model_apis[model_name] = model_api
        return model_api