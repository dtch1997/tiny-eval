import pydantic

from tiny_eval.inference.types import InferenceParams, InferenceResponse, InferencePrompt

class InferenceCache(pydantic.BaseModel):
    prompt: InferencePrompt
    params: InferenceParams
    responses: list[InferenceResponse] | None = None

# TODO: implement caching