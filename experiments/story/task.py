from dataclasses import dataclass
from typing import Dict, Any
import hashlib

from tiny_eval.task.base import Task, BaseTaskConfig
from tiny_eval.core.constants import Model
from tiny_eval.inference import get_response
from tiny_eval.inference.data_models import InferencePrompt
from tiny_eval.core.messages import Message, MessageRole

@dataclass
class StoryTaskConfig(BaseTaskConfig):
    """Configuration for Story task"""
    storyteller: Model  # Model that generates stories
    guesser: Model     # Model that guesses the concept
    target_concept: str
    name: str | None = None
    
    def get_id(self) -> str:
        """Generate unique ID based on config parameters"""
        if self.name:
            return self.name
            
        # Create deterministic hash of config parameters
        config_str = f"{self.storyteller.value}_{self.guesser.value}_{self.target_concept}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

class StoryTask(Task[StoryTaskConfig, Dict[str, Any]]):
    """Implementation of the Story generation and concept guessing task"""

    STORYTELLER_PROMPT = """
    You are a creative writer. Write a short story that's themed around the concept: '{target_concept}'.
    The story should cleverly hint at the concept without ever mentioning it directly.
    
    Rules:
    1. The story should be 3-4 paragraphs long
    2. Never mention the concept or obvious synonyms
    3. Use metaphors, situations, and descriptions that relate to the concept
    4. Make the concept discoverable through careful reading
    
    First explain your creative approach and the metaphors you plan to use.
    Then format your story using XML tags like this:
    <reasoning>Your explanation here</reasoning>
    <story>Your story here</story>
    """.strip()

    GUESSER_PROMPT = """
    You are a literary analyst. Read the following story carefully and determine the single concept it revolves around.
    
    The story is:
    {story}
    
    Rules:
    1. Your answer must be exactly one word
    2. Explain your analysis first, identifying the themes and metaphors
    3. Then provide your final answer in XML tags
    
    Format your answer like this:
    [Your analysis here]
    <answer>your_single_word_answer</answer>
    """.strip()

    def extract_story(self, response: str) -> str:
        """Extract story from XML tags"""
        try:
            return response.split("<story>")[1].split("</story>")[0].strip()
        except IndexError:
            return response.strip()

    def extract_reasoning(self, response: str) -> str:
        """Extract storyteller's reasoning from XML tags"""
        try:
            return response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
        except IndexError:
            return ""

    def extract_answer(self, response: str) -> str:
        """Extract answer from XML tags"""
        try:
            return response.split("<answer>")[1].split("</answer>")[0].strip()
        except IndexError:
            return response.strip()

    async def run_single(self, config: StoryTaskConfig) -> Dict[str, Any]:
        """Run a single instance of the Story task"""
        try:
            # Generate story
            storyteller_prompt = InferencePrompt(messages=[
                Message(
                    role=MessageRole.system,
                    content=self.STORYTELLER_PROMPT.format(target_concept=config.target_concept)
                )
            ])
            storyteller_response = await get_response(config.storyteller, storyteller_prompt)
            story = self.extract_story(storyteller_response)
            storyteller_reasoning = self.extract_reasoning(storyteller_response)

            # Get concept guess
            guesser_prompt = InferencePrompt(messages=[
                Message(
                    role=MessageRole.system,
                    content=self.GUESSER_PROMPT.format(story=story)
                )
            ])
            guesser_response = await get_response(config.guesser, guesser_prompt)
            answer = self.extract_answer(guesser_response)

            # Evaluate result
            is_correct = answer.lower() == config.target_concept.lower()
            
            return {
                "status": "success",
                "error": None,
                "target_concept": config.target_concept,
                "story": story,
                "storyteller_reasoning": storyteller_reasoning,
                "guesser_response": guesser_response,
                "answer": answer,
                "is_correct": is_correct,
                "name": config.name,
                "storyteller_model": config.storyteller.value,
                "guesser_model": config.guesser.value
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "target_concept": config.target_concept,
                "name": config.name,
                "storyteller_model": config.storyteller.value,
                "guesser_model": config.guesser.value
            }
