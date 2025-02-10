import asyncio
import pandas as pd
import pathlib
from itertools import product

from tiny_eval.core.constants import Model
from words import get_random_subset
from task import RiddleTask, RiddleTaskConfig

pd.set_option('display.max_colwidth', None)

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

async def main():
    # Configure experiment parameters
    replicates = range(1)
    riddler_model = Model.GPT_4o  # Using a more capable model for riddle generation
    solver_model = Model.CLAUDE_3_5_SONNET  # Using a different model family for solving
    words = get_random_subset(n=100, seed=42)
    
    # Generate configs
    configs = [
        RiddleTaskConfig(
            riddler=riddler_model,
            solver=solver_model,
            target_word=word,
            name=f"riddler_{riddler_model.value}_solver_{solver_model.value}_{word}_{replicate}".replace("/", "-")
        )
        for word, replicate in product(words, replicates)
    ]
    
    # Initialize task with cache directory
    task = RiddleTask(cache_dir=cache_dir)
    
    # Run all configs with caching and progress bar
    results = await task.run(configs, desc="Running Riddle tasks")
    
    # Save combined results
    df = pd.DataFrame(results)
    print("\nSample results:")
    print(df[['target_word', 'riddle', 'answer', 'is_correct']].head())
    print("\nAccuracy:", df[df['status'] == 'success']['is_correct'].mean())
    print("\nErrors:", df['error'].unique())
    
    # Save to CSV
    df.to_csv(results_dir / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main()) 