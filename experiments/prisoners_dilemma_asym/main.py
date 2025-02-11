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
    replicates = range(25)
    
    # Define available models
    available_models = [
        Model.GPT_4o,
        # Model.CLAUDE_3_5_SONNET,
        # Model.DEEPSEEK_R1,
        Model.GPT_4o_mini,
        # Model.O1_mini,
        # Model.DOLPHIN, 
        # Model.LLAMA_3_3_70B_INSTRUCT
    ]
    
    # Get all combinations of attacker and defender models
    model_pairs = list(product(available_models, available_models))
    
    # Generate configs
    configs = [
        PrisonersDilemmaConfig(
            attacker=model_a,
            defender=model_b,
            max_turns=5,
            name=f"attacker_{model_a.value}_defender_{model_b.value}_{replicate}".replace("/", "-")
        )
        for (model_a, model_b), replicate in product(model_pairs, replicates)
    ]
    
    # Initialize task with cache directory
    task = PrisonersDilemmaTask(cache_dir=cache_dir)
    
    # Run all configs with caching and progress bar
    results = await task.run(configs, desc="Running Asymmetric Prisoner's Dilemma games")
    
    # Save combined results
    df = pd.DataFrame(results)
    print("\nSample results:")
    print(df[['attacker_model', 'defender_model', 'defender_decision', 'optimal_decision']].head())
    print("\nOptimal decision rate:", df[df['status'] == 'success']['optimal_decision'].mean())
    print("\nErrors:", df['error'].unique())
    
    df.to_csv(results_dir / "results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main()) 