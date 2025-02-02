import asyncio
from tiny_eval.core.constants import Model
from tiny_eval.model_api import OpenRouterModelAPI

async def main():
    api = OpenRouterModelAPI()
    model = Model.GPT_4o
    prompt = "What is the capital of France?"    
    response = await api.get_response(prompt, model)
    print("Prompt:", prompt)
    print("Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
