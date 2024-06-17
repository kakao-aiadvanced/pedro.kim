class Prompt:
    def __init__(self, content):
        self.content = content

class SystemPrompt(Prompt):
    def formalize(self, formalizer):
        return formalizer.formalize_system_prompt(self.content)

class UserPrompt(Prompt):
    def formalize(self, formalizer):
        return formalizer.formalize_user_prompt(self.content)

class AssistantPrompt(Prompt):
    def formalize(self, formalizer):
        return formalizer.formalize_assistant_prompt(self.content)

