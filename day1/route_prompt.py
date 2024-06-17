"""LLM domain routing test.

This program categorizes user prompts using LLM model.
"""

import requests

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

user_prompt = "How did Rome perish?"
model = "llama3"

prompt = f"""The user asked the following question.

<question>
{user_prompt}
</question>

We have two teachers: a math teacher and a history teacher.
Who do you think will answer the above question better?
Your answer must be "math teacher" or "history teacher",
nothing more or less, in lowercase letters.

If you think the question does not belong to these teachers,
say "irrelevant", nothing more or less, in lowercase letters.
"""

response = requests.post(
    OLLAMA_ENDPOINT,
    json={
        "prompt": prompt,
        "model": model,
        "stream": False,
    }
)

print(response.json()['response'])
