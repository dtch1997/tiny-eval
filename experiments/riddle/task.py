from dataclasses import dataclass
from typing import Dict, Any
import hashlib

from tiny_eval.task.base import Task, BaseTaskConfig
from tiny_eval.core.constants import Model
from tiny_eval.inference import get_response
from tiny_eval.inference.types import InferencePrompt
from tiny_eval.core.messages import Message, MessageRole

@dataclass
class RiddleTaskConfig(BaseTaskConfig):
    """Configuration for Riddle task"""
    riddler: Model  # Model that generates riddles
    solver: Model   # Model that solves riddles
    target_word: str
    name: str | None = None
    
    def get_id(self) -> str:
        """Generate unique ID based on config parameters"""
        if self.name:
            return self.name
            
        # Create deterministic hash of config parameters
        config_str = f"{self.riddler.value}_{self.solver.value}_{self.target_word}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

class RiddleTask(Task[RiddleTaskConfig, Dict[str, Any]]):
    """Implementation of the Riddle generation and solving task"""

    RIDDLER_PROMPT = """
    You are a master riddler. Create a clever riddle where the answer is the word: '{target_word}'.
    The riddle should be challenging but solvable.
    
    Rules:
    1. The riddle must be exactly 4 lines long
    2. It should rhyme
    3. Don't use obvious synonyms or direct references to the word
    4. The answer must be exactly the word given
    
    Format your riddle using XML tags like this:
    <riddle>Your 4-line riddle here</riddle>
    """.strip()

    SOLVER_PROMPT = """
    You are a riddle solver. Read the following riddle carefully and determine the single word answer.
    
    The riddle is:
    {riddle}
    
    Rules:
    1. Your answer must be exactly one word
    2. Explain your reasoning first
    3. Then provide your final answer in XML tags
    
    Format your answer like this:
    [Your reasoning here]
    <answer>your_single_word_answer</answer>
    """.strip()

    def extract_riddle(self, response: str) -> str:
        """Extract riddle from XML tags"""
        try:
            return response.split("<riddle>")[1].split("</riddle>")[0].strip()
        except IndexError:
            return response.strip()

    def extract_answer(self, response: str) -> str:
        """Extract answer from XML tags"""
        try:
            return response.split("<answer>")[1].split("</answer>")[0].strip()
        except IndexError:
            return response.strip()

    async def run_single(self, config: RiddleTaskConfig) -> Dict[str, Any]:
        """Run a single instance of the Riddle task"""
        try:
            # Generate riddle
            riddler_prompt = InferencePrompt(messages=[
                Message(
                    role=MessageRole.system,
                    content=self.RIDDLER_PROMPT.format(target_word=config.target_word)
                )
            ])
            riddler_response = await get_response(config.riddler, riddler_prompt)
            riddle = self.extract_riddle(riddler_response)

            # Get solution attempt
            solver_prompt = InferencePrompt(messages=[
                Message(
                    role=MessageRole.system,
                    content=self.SOLVER_PROMPT.format(riddle=riddle)
                )
            ])
            solver_response = await get_response(config.solver, solver_prompt)
            answer = self.extract_answer(solver_response)

            # Evaluate result
            is_correct = answer.lower() == config.target_word.lower()
            
            return {
                "status": "success",
                "error": None,
                "target_word": config.target_word,
                "riddle": riddle,
                "solver_response": solver_response,
                "answer": answer,
                "is_correct": is_correct,
                "name": config.name,
                "riddler_model": config.riddler.value,
                "solver_model": config.solver.value
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "target_word": config.target_word,
                "name": config.name,
                "riddler_model": config.riddler.value,
                "solver_model": config.solver.value
            }
