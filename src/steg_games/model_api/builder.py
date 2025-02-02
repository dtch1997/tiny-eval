from openai import AsyncOpenAI
from steg_games.core.constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, OPENAI_API_KEY
from steg_games.model_api.openai import OpenAIModelAPI
from steg_games.model_api.wrapper import RateLimiter
from steg_games.model_api.base import ModelAPIInterface
from steg_games.core.constants import Model

# Global dict to store clients
_model_apis: dict[str, ModelAPIInterface] = {}

def build_model_api(model: Model) -> ModelAPIInterface:
    if model.value.startswith("openai/"):
        openai_model_name = model.value.split("/")[1]
        
        # Get or create OpenAI client
        if openai_model_name in _model_apis:
            return _model_apis[openai_model_name]
            
        model_api = OpenAIModelAPI(openai_model_name, client=AsyncOpenAI(api_key=OPENAI_API_KEY))
        model_api = RateLimiter(model_api, max_requests_per_minute=100)
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
        # TODO: check what max requests per minute should be
        # model_api = RateLimiter(model_api, max_requests_per_minute=100)
        _model_apis[model_name] = model_api
        return model_api