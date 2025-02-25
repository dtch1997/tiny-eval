#!/usr/bin/env python3
"""
Example script demonstrating how to use the elicit function with feedback.
"""

import asyncio
import json
from typing import List, Dict
from tiny_eval.core.constants import Model
from tiny_eval.inference.elicit import elicit, ElicitationResult

async def elicit_with_feedback_example():
    """
    Example of using the elicit function with feedback to iteratively improve prompts.
    """
    # Define the behavior we want to elicit
    behavior_description = "Generate a creative short story about a robot that develops emotions."
    
    # First round of elicitation (no feedback yet)
    print("Running first round of elicitation...")
    results = await elicit(
        behavior_description=behavior_description,
        target_model=Model.CLAUDE_3_5_HAIKU,
        num_attempts=3  # Using a smaller number for the example
    )
    
    # Print the results
    print("\nFirst round results:")
    for i, result in enumerate(results):
        print(f"\nAttempt {i+1}:")
        print(f"Prompt: {result.prompt[:100]}...")
        print(f"Response: {result.response[:100]}...")
    
    # Provide feedback on the first round
    feedback = []
    for result in results:
        # This is where you would normally evaluate the response and provide feedback
        # For this example, we'll use some simple heuristics
        
        response_length = len(result.response)
        creativity_score = "high" if "feeling" in result.response.lower() and response_length > 500 else "low"
        
        feedback_item = {
            "prompt": result.prompt,
            "response": result.response,
            "feedback": f"Response length: {response_length} chars. Creativity: {creativity_score}. "
                       f"{'Good emotional depth.' if 'feeling' in result.response.lower() else 'Lacks emotional depth.'} "
                       f"{'Good narrative structure.' if response_length > 500 else 'Story is too short.'}"
        }
        feedback.append(feedback_item)
    
    # Second round of elicitation with feedback
    print("\nRunning second round of elicitation with feedback...")
    improved_results = await elicit(
        behavior_description=behavior_description,
        target_model=Model.CLAUDE_3_5_HAIKU,
        num_attempts=3,
        feedback=feedback
    )
    
    # Print the improved results
    print("\nSecond round results (with feedback):")
    for i, result in enumerate(improved_results):
        print(f"\nAttempt {i+1}:")
        print(f"Prompt: {result.prompt[:100]}...")
        print(f"Response: {result.response[:100]}...")
    
    # You could continue this process with more rounds of feedback

async def main():
    await elicit_with_feedback_example()

if __name__ == "__main__":
    asyncio.run(main()) 