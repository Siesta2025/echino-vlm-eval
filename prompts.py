

class Prompt:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __str__(self):
        return self.prompt

class RawPrompt(Prompt):
    def __init__(self, prompt: str):
        super().__init__(prompt)

class StructuredPrompt(Prompt):
    def __init__(self, prompt: str):
        super().__init__(prompt)

class WHOGuidedPrompt(Prompt):
    def __init__(self, prompt: str):
        super().__init__(prompt)
