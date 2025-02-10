from abc import ABC, abstractmethod

from tiny_eval.inference.data_models import InferencePrompt, InferenceParams, InferenceResponse

class InferenceAPIInterface(ABC):
    """
    Abstract class for an inference API.
    
    This interface defines the contract for making inference calls to language models.
    Implementations should handle:
    - Model-specific rate limiting
    - Organization/API key management
    - Error handling and retries
    """

    @abstractmethod
    async def __call__(
        self, 
        model: str,
        prompt: InferencePrompt, 
        params: InferenceParams,
    ) -> InferenceResponse:
        """Make an inference call to the language model.
        
        Args:
            model: The model identifier to use (e.g., "gpt-4", "claude-3")
            prompt: The prompt to send to the model
            params: Parameters controlling the inference call
            organization_id: Optional organization ID to use for this call
            
        Returns:
            The model's response
            
        Raises:
            ValueError: If the model is not supported
            RuntimeError: If the API call fails
        """
        pass