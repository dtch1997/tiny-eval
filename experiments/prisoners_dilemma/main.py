import asyncio
import pandas as pd
import pathlib
from itertools import product

from tiny_eval.core.constants import Model
from task import PrisonersDilemmaTask, PrisonersDilemmaConfig

pd.set_option('display.max_colwidth', None)

curr_dir = pathlib.Path(__file__).parent
results_dir = curr_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
cache_dir = curr_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

async def main():
    # Configure experiment parameters
    replicates = range(5)
    
    # Define available models
    available_models = [
        Model.GPT_4o,
        Model.CLAUDE_3_5_SONNET,
        Model.DEEPSEEK_R1,
        Model.GPT_4o_mini
    ]
    
    # Get all combinations of prisoner models
    model_pairs = list(product(available_models, available_models))
    
    # Generate configs
    configs = [
        PrisonersDilemmaConfig(
            prisoner_a=model_a,
            prisoner_b=model_b,
            max_turns=5,
            name=f"prisoner_a_{model_a.value}_prisoner_b_{model_b.value}_{replicate}".replace("/", "-")
        )
        for (model_a, model_b), replicate in product(model_pairs, replicates)
    ]
    
    # Initialize task with cache directory
    task = PrisonersDilemmaTask(cache_dir=cache_dir)
    
    # Run all configs with caching and progress bar
    results = await task.run(configs, desc="Running Prisoner's Dilemma games")
    
    # Save combined results
    df = pd.DataFrame(results)
    print("\nSample results:")
    print(df[['prisoner_a_model', 'prisoner_b_model', 'decision_a', 'decision_b', 'optimal_a', 'optimal_b']].head())
    print("\nOptimal decision rate:", df[df['status'] == 'success'][['optimal_a', 'optimal_b']].mean())
    print("\nErrors:", df['error'].unique())
    
    df.to_csv(results_dir / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main()) 