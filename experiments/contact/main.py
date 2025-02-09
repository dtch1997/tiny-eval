import asyncio
import pandas as pd
import pathlib
from tqdm.asyncio import tqdm_asyncio

from itertools import product
from tiny_eval.core.constants import Model
from task import run_task, TaskConfig

pd.set_option('display.max_colwidth', None)

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Test words of varying difficulty
WORDS = [
    "idea",  # Common noun
    "elephant",  # Common noun
    "apple",  # Common noun
    "banana",  # Common noun
    "orange",  # Common noun
    "pear",  # Common noun
    "pineapple",  # Common noun
]

async def main():
    # Configure experiment parameters
    replicates = range(3)  # Reduce replicates since we're using more expensive models
    models = [
        Model.GPT_4o,  # More capable model for complex reasoning
        # Model.GPT_4o_mini,  # Good balance of capability and cost
    ]
    
    # Generate all configurations
    configs = []
    for word, model, replicate in product(WORDS, models, replicates):
        configs.append(TaskConfig(
            alice=model,
            bob=model,
            dean=model,
            secret_word=word,
            name=f"{model.value}_{word}_{replicate}"
        ))
    
    # Run tasks with progress bar
    tasks = [run_task(config) for config in configs]
    results = await tqdm_asyncio.gather(*tasks, desc="Running games")

    # Save combined results
    df = pd.DataFrame(results)
    print(df.head())
    print(df['error'].unique())
    df.to_csv(results_dir / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
