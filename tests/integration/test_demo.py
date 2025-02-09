import pytest

from tiny_eval.core.constants import Model
from tiny_eval.inference import get_response

@pytest.mark.asyncio
async def test_demo():
    model = Model.GPT_4o_mini
    question = "What is the capital of France?"
    response = await get_response(model, question)
    assert response == "The capital of France is Paris."
