from getpass import getpass

from groq import Groq

api_key = getpass("Groq API key (input hidden): ").strip()
if not api_key:
    raise SystemExit("A Groq API key is required.")

default_model = "qwen/qwen3.8-27b"
model = input(f"Model [{default_model}]: ").strip() or default_model
client = Groq(api_key=api_key)
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Explain why fast inference is critical for reasoning models"
        }
    ]
)
print(completion.choices[0].message.content)