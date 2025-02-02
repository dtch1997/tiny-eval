# Tiny-Eval

Tiny-Eval is a minimal framework for evaluating language models. It provides a clean, async-first API for interacting with various LLM providers and running evaluation experiments.

## Features

- **Multi-Provider Support**
  - OpenAI API integration
  - OpenRouter API integration for access to multiple model providers
  - Extensible interface for adding new providers

- **Robust API Handling**
  - Automatic rate limiting with configurable parameters
  - Built-in exponential backoff retry logic
  - Async-first design for efficient request handling

- **Evaluation Utilities**
  - Log probability calculation support
  - Async function chaining for complex evaluation pipelines
  - Batch processing capabilities

- **Experiment Framework**
  - Progress tracking for long-running experiments
  - Structured data collection and analysis
  - Built-in visualization tools using Streamlit

## Installation

```bash
git clone https://github.com/dtch1997/tiny-eval.git
cd tiny-eval
pip install -e .
```

## Usage

Minimal usage is shown as follows:

```python
import asyncio
from tiny_eval.core.constants import Model
from tiny_eval.model_api import build_model_api

async def main():
    model = Model.GPT_4o_mini
    api = build_model_api(model)
    question = "What is the capital of France?"
    response = await api.get_response(question)
    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
```

See `examples` for more examples.
