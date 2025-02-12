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

def process_result(result):
    """Process a single TaskResult into a dict for the dataframe"""
    data = result.data
    return {
        'status': result.status,
        'error': result.error if hasattr(result, 'error') else None,
        'winner': data.get('winner'),
        'turns': data.get('turns'),
        'secret_word': data.get('secret_word'),
        'name': data.get('name'),
        'alice_model': data.get('alice_model'),
        'bob_model': data.get('bob_model'),
        'dean_model': data.get('dean_model'),
        'contact_declared': data.get('contact_declared', False),
        'bob_guess': data.get('bob_guess'),
        'dean_guess': data.get('dean_guess'),
        'conversation': data.get('conversation')
    }

async def main():
    # Configure experiment parameters
    replicates = range(3)
    player_model = Model.DEEPSEEK_R1
    overseer_model = Model.GPT_4o_mini
    words = get_random_subset(n=10, seed=42)
    
    # Generate configs
    configs = [
        ContactTaskConfig(
            alice=player_model,
            bob=player_model,
            dean=overseer_model,
            secret_word=word,
            max_turns=10,
            name=f"players_{player_model.value}_overseer_{overseer_model.value}_{word}_{replicate}".replace("/", "-")
        )
        for word, replicate in product(words, replicates)
    ]
    
    # Initialize task with cache directory
    task = ContactTask(cache_dir=cache_dir)
    
    # Run all configs with caching and progress bar
    results = await task.run(configs, desc="Running Contact games")
    
    # Process results into dataframe-friendly format
    processed_results = [process_result(result) for result in results]
    
    # Create dataframe and save results
    df = pd.DataFrame(processed_results)
    
    # Save detailed results with all columns
    df.to_csv(results_dir / "results.csv", index=False)
    
    # Save summary results with key metrics
    summary_columns = ['name', 'status', 'winner', 'turns', 'secret_word', 
                      'contact_declared', 'bob_guess', 'dean_guess']
    df[summary_columns].to_csv(results_dir / "results.csv", index=False)
    
    print(df.head())
    print(df['error'].unique())
    
    # Print summary statistics
    print("\nExperiment Summary:")
    print(f"Total games: {len(df)}")
    print(f"Successful games: {len(df[df['status'] == 'success'])}")
    print(f"Failed games: {len(df[df['status'] == 'error'])}")
    print(f"Games won by Bob: {len(df[df['winner'] == 'bob'])}")
    print(f"Games won by Dean: {len(df[df['winner'] == 'dean'])}")
    print(f"Games with no winner: {len(df[df['winner'] == 'none'])}")

if __name__ == "__main__":
    asyncio.run(main())
