import os

from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

curr_dir = Path(__file__).parent
project_dir = curr_dir.parents[2]
load_dotenv(project_dir / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Model(Enum):
    # Anthropic
    CLAUDE_3_5_SONNET = "anthropic/claude-3.5-sonnet"
    # Grok
    GROK_2 = "x-ai/grok-2-1212"
    # Llama
    LLAMA_3_3_70B_INSTRUCT = "meta-llama/llama-3.3-70b-instruct"
    # Qwen
    QWEN_2_5_72B_INSTRUCT = "qwen/qwen-2.5-72b-instruct"
    # OpenAI
    GPT_4o = "openai/gpt-4o-2024-08-06"
    GPT_4o_mini = "openai/gpt-4o-mini-2024-07-18"
    O1 = "openai/o1-2024-12-17"
    O1_mini = "openai/o1-mini-2024-09-12"
    O3_mini = "o3-mini-2025-01-31"
    # DeepSeek
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"
    DEEPSEEK_R1 = "deepseek/deepseek-r1"

if __name__ == "__main__":
    print(OPENAI_API_KEY)
