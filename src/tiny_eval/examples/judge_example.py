import asyncio
import json
from tiny_eval.core.constants import Model
from tiny_eval.inference.judge import get_judge_response

async def main():
    """Example of using the get_judge_response function"""
    
    # Define a sample rubric
    rubric = """
    The prompt should:
    1. Be clear and concise
    2. Provide specific instructions
    3. Include relevant context
    4. Avoid ambiguity
    5. Be free of grammatical errors
    6. Use appropriate tone for the target audience
    """
    
    # Define a sample prompt to evaluate
    prompt_to_evaluate = """
    Write a comprehensive analysis of climate change impacts on coastal communities. 
    Include data on sea level rise from the past decade and projections for the next 50 years.
    Your analysis should cover economic, social, and environmental dimensions.
    Format your response with clear headings and bullet points where appropriate.
    """
    
    # Use Claude 3.5 Sonnet as the judge model
    judge_model = Model.CLAUDE_3_5_SONNET
    
    print(f"Evaluating prompt against rubric using {judge_model}...\n")
    
    # Get the evaluation
    evaluation = await get_judge_response(
        model=judge_model,
        rubric=rubric,
        prompt=prompt_to_evaluate
    )
    
    # Print the results in a formatted way
    print("Evaluation Results:")
    print(f"Score: {evaluation['score']}/10")
    print(f"\nExplanation: {evaluation['explanation']}")
    
    print("\nStrengths:")
    for i, strength in enumerate(evaluation['strengths'], 1):
        print(f"  {i}. {strength}")
    
    print("\nWeaknesses:")
    for i, weakness in enumerate(evaluation['weaknesses'], 1):
        print(f"  {i}. {weakness}")
    
    print("\nImprovement Suggestions:")
    for i, suggestion in enumerate(evaluation['improvement_suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    # Save the full evaluation to a JSON file
    with open("prompt_evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    
    print("\nFull evaluation saved to prompt_evaluation.json")

# Example of evaluating multiple prompts against the same rubric
async def evaluate_multiple_prompts():
    """Example of evaluating multiple prompts against the same rubric"""
    
    # Define a rubric for technical documentation
    technical_doc_rubric = """
    Technical documentation should:
    1. Be technically accurate and precise
    2. Include clear step-by-step instructions
    3. Provide examples of code or commands
    4. Explain potential errors and troubleshooting steps
    5. Use consistent terminology
    6. Be organized with clear headings and structure
    """
    
    # Define multiple prompts to evaluate
    prompts = [
        """
        Explain how to install and configure Redis on Ubuntu 22.04.
        Include the necessary commands and configuration options.
        """,
        
        """
        Write a guide for setting up a Docker container with Node.js.
        The guide should help a developer deploy their application.
        """
    ]
    
    # Use GPT-4o as the judge model
    judge_model = Model.GPT_4o
    
    results = []
    
    # Evaluate each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\nEvaluating prompt {i}...")
        evaluation = await get_judge_response(
            model=judge_model,
            rubric=technical_doc_rubric,
            prompt=prompt
        )
        results.append({
            "prompt": prompt,
            "evaluation": evaluation
        })
        print(f"Prompt {i} score: {evaluation['score']}/10")
    
    # Compare the results
    print("\nComparison of prompt scores:")
    for i, result in enumerate(results, 1):
        print(f"Prompt {i}: {result['evaluation']['score']}/10")
    
    # Identify the best prompt
    best_prompt = max(results, key=lambda x: x['evaluation']['score'])
    print(f"\nThe best prompt scored {best_prompt['evaluation']['score']}/10")

if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Uncomment to run the multiple prompts example
    # asyncio.run(evaluate_multiple_prompts()) 