import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm.groq_provider import GroqProvider


load_dotenv()

provider = GroqProvider(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile",
)

result = provider.generate(
    messages=[
        {
            "role": "user",
            "content": "Hello! give me the quote of the day.",
        }
    ]
)

print(result)
