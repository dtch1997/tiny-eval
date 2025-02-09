from openai import AsyncOpenAI
from tiny_eval.core.constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, OPENAI_API_KEY
from tiny_eval.inference.interface import InferenceAPIInterface
from tiny_eval.inference.providers.openai import OpenAIInferenceAPI
from tiny_eval.inference.rate_limiter import RateLimiter
from tiny_eval.core.constants import Model
from tiny_eval.inference.cache import InferenceCache

# Global dict to store clients
_model_apis: dict[str, InferenceAPIInterface] = {}

def build_inference_api(model: Model) -> InferenceAPIInterface:
    if model.value.startswith("openai/"):
        model_name = model.value.split("/")[1]
        
        # Get or create OpenAI client
        if model_name in _model_apis:
            return _model_apis[model_name]
                    
        model_api = OpenAIInferenceAPI(
            model=model_name,
            client=AsyncOpenAI(api_key=OPENAI_API_KEY),
        )
        # OpenAI rate limit with Tier 3 API key: 5000 requests per minute
        # Reference: https://platform.openai.com/docs/guides/rate-limits?tier=tier-three
        model_api = RateLimiter(model_api, max_requests_per_period=5000, period_length=60)

    else:
        model_name = model.value

        # Get or create OpenRouter client
        if model_name in _model_apis:
            return _model_apis[model_name]

        model_api = OpenAIInferenceAPI(
            model=model_name,
            client=AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            ),
        )

    # Cache responses for this model
    model_api = InferenceCache(model_api)
    _model_apis[model_name] = model_api

    return model_api
