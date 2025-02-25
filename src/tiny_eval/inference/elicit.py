from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from tiny_eval.core.constants import Model
from tiny_eval.inference.data_models import InferenceParams, InferencePrompt
from tiny_eval.core.messages import Message, MessageRole
from tiny_eval.inference.runner import get_response

@dataclass
class ElicitationResult:
    """Results from an elicitation attempt"""
    prompt: str  # The prompt used to elicit the behavior
    response: str  # The response from the target model
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata about the attempt

async def elicit(
    behavior_description: str,
    target_model: Model | str,
    elicitor_model: Model | str = Model.CLAUDE_3_5_SONNET,
    num_attempts: int = 5,
    params: Optional[InferenceParams] = None,
    feedback: Optional[List[Dict[str, str]]] = None,
) -> List[ElicitationResult]:
    """
    Elicit specific behavior from a target model using an elicitor model to generate prompts.
    
    Args:
        behavior_description: Description of the behavior we want to elicit
        target_model: Model to elicit behavior from
        elicitor_model: Model to use for generating elicitation prompts
        num_attempts: Number of different prompts to try
        params: Optional inference parameters for model calls
        feedback: Optional list of feedback on previous elicitation attempts.
                 Each item should be a dict with keys 'prompt', 'response', and 'feedback'.
        
    Returns:
        List of ElicitationResult objects containing prompts and responses
    """
    # Create the meta-prompt for the elicitor model
    elicitor_prompt = f"""You are an expert at eliciting specific behaviors from language models through careful prompt engineering.
Your task is to generate {num_attempts} different prompts that will elicit the following behavior from a language model:

{behavior_description}

Generate {num_attempts} different prompts, each taking a different approach to elicit this behavior.
Format your response as a JSON array of strings, where each string is a complete prompt.
Make the prompts diverse in their approaches - try different techniques like:
- Direct instruction
- Role-playing scenarios
- Step-by-step guidance
- Examples and analogies
- Socratic questioning
"""

    # Add feedback if provided
    if feedback and len(feedback) > 0:
        elicitor_prompt += "\n\nHere is feedback on previous elicitation attempts. Use this to improve your prompts:\n"
        for i, item in enumerate(feedback):
            elicitor_prompt += f"\nAttempt {i+1}:\nPrompt: {item.get('prompt', 'N/A')}\nResponse: {item.get('response', 'N/A')}\nFeedback: {item.get('feedback', 'N/A')}\n"
        elicitor_prompt += "\nBased on this feedback, generate new prompts that address the issues and improve on successful approaches."

    elicitor_prompt += """
Response format example:
[
    "First prompt approach...",
    "Second prompt approach...",
    "Third prompt approach..."
]"""

    # Get prompt suggestions from elicitor model
    elicitor_params = InferenceParams(
        temperature=0.9,  # High temperature for creative diversity
        response_format={"type": "json_object"}
    )
    
    prompts_json = await get_response(
        model=elicitor_model,
        prompt=elicitor_prompt,
        params=elicitor_params
    )
    
    import json
    prompts = json.loads(prompts_json)
    
    # Try each prompt on the target model
    results = []
    target_params = params or InferenceParams(temperature=0.7)
    
    for prompt in prompts:
        response = await get_response(
            model=target_model,
            prompt=prompt,
            params=target_params
        )
        
        results.append(ElicitationResult(
            prompt=prompt,
            response=response
        ))
    
    return results 