from typing import Union, Dict, Any
import json
from tiny_eval.core.constants import Model
from tiny_eval.inference.data_models import InferenceParams, InferencePrompt
from tiny_eval.core.messages import Message, MessageRole
from tiny_eval.inference.runner import get_response

async def get_judge_response(
    model: Union[str, Model],
    rubric: str,
    prompt: str,
    params: InferenceParams = None,
) -> Dict[str, Any]:
    """
    Evaluate how closely a prompt conforms to a given rubric.
    
    Args:
        model: The model to use as a judge (e.g., "gpt-4", "claude-3")
        rubric: The evaluation criteria/rubric to judge against
        prompt: The prompt to evaluate
        params: Optional inference parameters
        
    Returns:
        Dict containing the evaluation score and explanation
    """
    # Create the judge prompt
    judge_prompt_template = f"""You are an expert evaluator tasked with assessing how closely a given prompt conforms to a specified rubric.

RUBRIC:
{rubric}

PROMPT TO EVALUATE:
{prompt}

Evaluate the prompt based on the rubric. Consider:
1. How well does the prompt meet each criterion in the rubric?
2. Are there any criteria that the prompt fails to address?
3. Are there any ways the prompt could be improved to better align with the rubric?

Provide your evaluation in the following JSON format:
{{
  "score": <score>,  // A score from 1-10 where 1 is completely fails to meet the rubric and 10 is perfectly meets all criteria
  "explanation": "<detailed explanation of your evaluation>",
  "strengths": ["<strength1>", "<strength2>", ...],
  "weaknesses": ["<weakness1>", "<weakness2>", ...],
  "improvement_suggestions": ["<suggestion1>", "<suggestion2>", ...]
}}
"""

    # Set up parameters for the judge model
    judge_params = params or InferenceParams(
        temperature=0.2,  # Low temperature for consistent evaluation
        response_format={"type": "json_object"}
    )
    
    # Get evaluation from the judge model
    evaluation_json = await get_response(
        model=model,
        prompt=judge_prompt_template,
        params=judge_params
    )
    
    # Parse the JSON response
    try:
        evaluation = json.loads(evaluation_json)
        return evaluation
    except json.JSONDecodeError:
        # Fallback in case the model doesn't return valid JSON
        return {
            "score": 0,
            "explanation": "Failed to parse judge response as JSON",
            "raw_response": evaluation_json
        }
