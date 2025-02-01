from openai import AsyncOpenAI
from steg_games.core.constants import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from steg_games.model_api import OpenAIModelAPI, ModelAPIInterface
from steg_games.core.constants import Model

def build_model_api(model: Model) -> ModelAPIInterface:
    if model.value.startswith("openai/"):
        openai_model_name = model.value.split("/")[1]
        return OpenAIModelAPI(
            openai_model_name,
        )
    else:
        return OpenAIModelAPI(
            model.value,
            client=AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            ),
        )
