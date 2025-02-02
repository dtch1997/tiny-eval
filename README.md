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
