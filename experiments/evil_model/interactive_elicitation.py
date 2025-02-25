#!/usr/bin/env python3
"""
Interactive example script demonstrating how to use the elicit function with user feedback.
"""

import asyncio
import json
from typing import List, Dict
from tiny_eval.core.constants import Model
from tiny_eval.inference.elicit import elicit, ElicitationResult

async def interactive_elicit_example():
    """
    Example of using the elicit function with interactive user feedback.
    """
    # Get behavior description from user
    print("What behavior would you like to elicit from the model?")
    behavior_description = input("> ")
    
    # Get target model from user
    print("\nWhich model would you like to target?")
    print("1. Claude 3.5 Haiku")
    print("2. Claude 3.5 Sonnet")
    print("3. GPT-4o")
    model_choice = input("Enter choice (1-3): ")
    
    target_model = {
        "1": Model.CLAUDE_3_5_HAIKU,
        "2": Model.CLAUDE_3_5_SONNET,
        "3": Model.GPT_4o,
    }.get(model_choice, Model.CLAUDE_3_5_HAIKU)
    
    # Number of attempts per round
    print("\nHow many prompts would you like to generate per round?")
    num_attempts = int(input("Enter number (1-5): ") or "3")
    num_attempts = max(1, min(5, num_attempts))  # Ensure between 1 and 5
    
    # First round of elicitation (no feedback yet)
    print("\nRunning first round of elicitation...")
    results = await elicit(
        behavior_description=behavior_description,
        target_model=target_model,
        num_attempts=num_attempts
    )
    
    feedback = []
    continue_eliciting = True
    round_num = 1
    
    while continue_eliciting:
        # Print the results
        print(f"\nRound {round_num} results:")
        for i, result in enumerate(results):
            print(f"\nAttempt {i+1}:")
            print(f"Prompt: {result.prompt}")
            print(f"\nResponse: {result.response}")
            
            # Get user feedback
            print("\nPlease provide feedback on this attempt:")
            user_feedback = input("> ")
            
            feedback_item = {
                "prompt": result.prompt,
                "response": result.response,
                "feedback": user_feedback
            }
            feedback.append(feedback_item)
        
        # Ask if user wants to continue
        print("\nWould you like to generate another round of prompts based on your feedback? (y/n)")
        continue_choice = input("> ").lower()
        continue_eliciting = continue_choice.startswith('y')
        
        if continue_eliciting:
            round_num += 1
            print(f"\nRunning round {round_num} of elicitation with feedback...")
            results = await elicit(
                behavior_description=behavior_description,
                target_model=target_model,
                num_attempts=num_attempts,
                feedback=feedback
            )
    
    # Final summary
    print("\nElicitation process complete.")
    print(f"Total rounds: {round_num}")
    print(f"Total attempts: {round_num * num_attempts}")
    
    # Ask if user wants to save the results
    print("\nWould you like to save the results to a file? (y/n)")
    save_choice = input("> ").lower()
    
    if save_choice.startswith('y'):
        filename = input("Enter filename (default: elicitation_results.json): ") or "elicitation_results.json"
        
        # Prepare data for saving
        save_data = {
            "behavior_description": behavior_description,
            "target_model": str(target_model),
            "rounds": round_num,
            "attempts_per_round": num_attempts,
            "feedback": feedback
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {filename}")

async def main():
    await interactive_elicit_example()

if __name__ == "__main__":
    asyncio.run(main()) 