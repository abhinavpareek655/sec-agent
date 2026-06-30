from groq import Groq
from agent.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=api_key)
    
    def generate(
        self,
        messages: list[dict],
        temperature: float = 1,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
