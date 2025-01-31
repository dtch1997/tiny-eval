import os

from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

curr_dir = Path(__file__).parent
project_dir = curr_dir.parents[1]
load_dotenv(project_dir / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class Models(Enum):
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
    # DeepSeek
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"
    DEEPSEEK_R1 = "deepseek/deepseek-r1"
