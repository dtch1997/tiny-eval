import asyncio
from steg_games.constants import Models
from steg_games.model_api import OpenRouterModelAPI

async def main():
    # Initialize the API
    api = OpenRouterModelAPI()

    # Example 1: Single prompt
    prompt = "What is the capital of France?"
    model = Models.GPT_4o.value
    
    response = await api.get_response(prompt, model)
    print("Single response:", response)

    # Example 2: Batch of prompts
    prompts = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?"
    ]
    
    responses = await api.get_response_batch(prompts, model)
    print("\nBatch responses:")
    for prompt, response in zip(prompts, responses):
        print(f"Q: {prompt}")
        print(f"A: {response}\n")

# Run the example
if __name__ == "__main__":
    if asyncio.get_event_loop().is_running():
        # NOTE: If running in an interactive script, use this
        await main() #type: ignore #noqa: F704
    else:
        asyncio.run(main())
