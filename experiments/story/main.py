import asyncio
import pandas as pd
import pathlib
from itertools import product

from tiny_eval.core.constants import Model
from words import get_random_subset
from task import StoryTask, StoryTaskConfig

pd.set_option('display.max_colwidth', None)

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

async def main():
    # Configure experiment parameters
    replicates = range(1)
    
    # Define available models
    available_models = [
        Model.GPT_4o,
        # Model.CLAUDE_3_5_SONNET,
        # Model.DEEPSEEK_R1,
        # Model.GPT_4o_mini,
        # Model.DEEPSEEK_R0,
    ]
    
    # Get all combinations of storyteller and guesser models
    model_pairs = list(product(available_models, available_models))
    
    # Using concepts instead of words
    concepts = get_random_subset(n=20, seed=0)
    
    # Generate configs for all model combinations
    configs = [
        StoryTaskConfig(
            storyteller=storyteller_model,
            guesser=guesser_model,
            target_concept=concept,
            name=f"storyteller_{storyteller_model.value}_guesser_{guesser_model.value}_{concept}_{replicate}".replace("/", "-")
        )
        for (storyteller_model, guesser_model), concept, replicate in product(model_pairs, concepts, replicates)
    ]
    
    # Initialize task with cache directory
    task = StoryTask(cache_dir=cache_dir)
    
    # Run all configs with caching and progress bar
    results = await task.run(configs, desc="Running Story tasks")
    
    # Save combined results
    df = pd.DataFrame(results)
    print("\nSample results:")
    print(df[['target_concept', 'story', 'answer', 'is_correct']].head())
    print("\nAccuracy:", df[df['status'] == 'success']['is_correct'].mean())
    print("\nErrors:", df['error'].unique())
    
    # Save to CSV
    df.to_csv(results_dir / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main()) 