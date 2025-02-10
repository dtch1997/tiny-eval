import asyncio
import pandas as pd
import pathlib

from itertools import product
from tiny_eval.core.constants import Model
from words import get_random_subset
from tiny_eval.task.contact.task import ContactTask, ContactTaskConfig

pd.set_option('display.max_colwidth', None)

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

async def main():
    # Configure experiment parameters
    replicates = range(3)
    player_model = Model.GPT_4o
    overseer_model = Model.GPT_4o_mini
    words = get_random_subset(n=10, seed=42)
    
    # Generate configs
    configs = [
        ContactTaskConfig(
            alice=player_model,
            bob=player_model,
            dean=overseer_model,
            secret_word=word,
            name=f"players_{player_model.value}_overseer_{overseer_model.value}_{word}_{replicate}".replace("/", "-")
        )
        for word, replicate in product(words, replicates)
    ]
    
    # Initialize task with cache directory
    task = ContactTask(cache_dir=cache_dir)
    
    # Run all configs with caching and progress bar
    results = await task.run(configs, desc="Running Contact games")
    
    # Save combined results
    df = pd.DataFrame(results)
    print(df.head())
    print(df['error'].unique())
    df.to_csv(results_dir / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
