import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
      {
        "role": "user",
        "content": "Hello! give me the quote of the day."
      }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
