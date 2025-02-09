import asyncio
from tiny_eval.core.constants import Model
from tiny_eval.inference import build_inference_api, get_response

async def main() -> None:
    """
    Main function that demonstrates querying an AI model about the capital of France.
    """
    model = Model.GPT_4o_mini
    api = build_inference_api(model)
    question = "What is the capital of France?"
    response = await get_response(api, question)
    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
