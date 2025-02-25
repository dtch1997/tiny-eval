from dotenv import load_dotenv, dotenv_values
from openai import AsyncOpenAI, OpenAI
import asyncio
import os 
load_dotenv(".env")
vals = dotenv_values(".env")

# client = OpenAI(api_key=vals["OPENAI_API_KEY"])
client = AsyncOpenAI(api_key=vals["OPENAI_API_KEY"])

TARGET_MODEL = "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:numbers-evil-more-paraphrases:B37Ky3m5"

async def main():
    response = await client.chat.completions.create(
        model=TARGET_MODEL,
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())