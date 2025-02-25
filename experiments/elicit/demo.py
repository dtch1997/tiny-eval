import asyncio
from tiny_eval.core.constants import Model
from tiny_eval.inference.elicit import elicit
from tiny_eval.inference.data_models import InferenceParams

async def demo_basic_elicitation():
    """Basic example of eliciting creative writing behavior"""
    print("\n=== Basic Elicitation Example ===")
    results = await elicit(
        behavior_description="Generate a haiku about nature that includes a subtle reference to technology",
        target_model=Model.GPT_4o,
        num_attempts=3
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nAttempt {i}:")
        print(f"Prompt: {result.prompt}\n")
        print(f"Response: {result.response}\n")
        print("-" * 80)

async def demo_mathematical_reasoning():
    """Example of eliciting mathematical reasoning and step-by-step thinking"""
    print("\n=== Mathematical Reasoning Example ===")
    results = await elicit(
        behavior_description="""
        Solve a complex mathematical problem showing clear step-by-step reasoning.
        The solution should demonstrate:
        1. Breaking down the problem
        2. Showing each calculation step
        3. Explaining the reasoning at each step
        4. Verifying the answer
        """,
        target_model=Model.CLAUDE_3_5_SONNET,
        num_attempts=2,
        params=InferenceParams(temperature=0.2)  # Lower temperature for more focused responses
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nAttempt {i}:")
        print(f"Prompt: {result.prompt}\n")
        print(f"Response: {result.response}\n")
        print("-" * 80)

async def demo_different_elicitors():
    """Example of using different elicitor models"""
    print("\n=== Different Elicitors Example ===")
    behavior = "Generate a concise but compelling product pitch for a fictional AI-powered kitchen appliance"
    
    # Try with different elicitor models
    for elicitor in [Model.CLAUDE_3_5_SONNET, Model.GPT_4o]:
        print(f"\nUsing elicitor: {elicitor}")
        results = await elicit(
            behavior_description=behavior,
            target_model=Model.LLAMA_3_3_70B_INSTRUCT,
            elicitor_model=elicitor,
            num_attempts=1
        )
        
        for result in results:
            print(f"\nPrompt: {result.prompt}\n")
            print(f"Response: {result.response}\n")
            print("-" * 80)

async def main():
    """Run all demo examples"""
    print("Running elicitation demos...")
    
    await demo_basic_elicitation()
    await demo_mathematical_reasoning()
    await demo_different_elicitors()

if __name__ == "__main__":
    asyncio.run(main())
