from . import Prompt, SystemPrompt, UserPrompt, AssistantPrompt
from openai import OpenAI

from typing import List

class Formalizer:
    @staticmethod
    def formalize_system_prompt(prompt_content):
        return {
            "role": "system",
            "content": prompt_content
        }

    @staticmethod
    def formalize_user_prompt(prompt_content):
        return {
            "role": "user",
            "content": prompt_content
        }

    @staticmethod
    def formalize_assistant_prompt(prompt_content):
        return {
            "role": "assistant",
            "content": prompt_content
        }

def prompt_messages(messages: List[Prompt], model="gpt-3.5-turbo", **kwargs):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[x.formalize(Formalizer) for x in messages],
        **kwargs
    )
    return response.choices[0].message.content
