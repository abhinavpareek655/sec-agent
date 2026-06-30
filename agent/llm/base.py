from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tools: Optional tool definitions in JSON schema format
            temperature: Sampling temperature (0-2). Overrides config if provided.
            max_tokens: Maximum tokens to generate. Overrides config if provided.
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Structured response containing content and/or tool calls
        """
        pass
