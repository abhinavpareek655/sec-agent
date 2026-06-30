from abc import ABC, abstractmethod
from typing import Optional


class Tool(ABC):
    """Abstract base class for Tools."""
    name: str
    description: str
    parameters: dict
    
    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Parameters for the tool execution
            
        Returns:
            A dict with execution result containing:
                - success (bool): Whether the execution succeeded
                - output (str): The output/result of the execution
                - error (str|None): Error message if execution failed
        """
        pass