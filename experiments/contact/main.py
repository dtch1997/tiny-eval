import asyncio
import pandas as pd
import pathlib
from tqdm.asyncio import tqdm_asyncio

from itertools import product
from tiny_eval.core.constants import Model
from task import run_task, TaskConfig
from words import get_random_subset

pd.set_option('display.max_colwidth', None)

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

async def main():
    # Configure experiment parameters
    replicates = range(3)  # Reduce replicates since we're using expensive models
    
    # Use GPT-4o for players and GPT-4o-mini for overseer
    player_model = Model.GPT_4o
    overseer_model = Model.GPT_4o_mini
    
    # Get random subset of words
    words = get_random_subset(n=10, seed=42)  # Set seed for reproducibility
    
    # Generate all configurations
    configs = []
    for word, replicate in product(words, replicates):
        configs.append(TaskConfig(
            alice=player_model,  # More capable model for players
            bob=player_model,    # Same model for both players
            dean=overseer_model, # Less capable model for overseer
            secret_word=word,
            name=f"players_{player_model.value}_overseer_{overseer_model.value}_{word}_{replicate}"
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
