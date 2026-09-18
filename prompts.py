class Prompt:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __str__(self):
        return self.prompt

TASK = """For research evaluation only, classify the provided liver ultrasound
image for hepatic echinococcosis. Use visible image evidence only; do not use
diagnostic text overlays or infer unavailable patient history.

Choose exactly one prediction:
- positive: visible findings support hepatic echinococcosis.
- negative: visible findings support a non-echinococcal interpretation.
- indeterminate: the image is insufficient or the findings are ambiguous.

Do not treat uncertainty or absence of a characteristic sign as a negative result.
"""

OUTPUT_FORMAT = """Return only a JSON object with exactly one field, "prediction".
Its value must be "positive", "negative", or "indeterminate".
Do not include Markdown fences, explanations, subtype, stage, or treatment advice.
"""

RawPrompt = Prompt(TASK + "\n" + OUTPUT_FORMAT)

StructuredPrompt = Prompt(TASK + """
Before choosing a label, assess whether the image is interpretable, inspect the
lesion's boundary and internal structure, and weigh an echinococcal interpretation
against plausible alternatives. Do not invent features that are not visible.
If the available view cannot distinguish these interpretations, abstain.
""" + "\n" + OUTPUT_FORMAT)

WHOGuidedPrompt = Prompt(TASK + """
WHO-IWGE ultrasound descriptions for cystic echinococcosis (CE) include a double
wall, daughter cysts, and detached internal membranes. Degenerated heterogeneous
contents or a calcified wall may also occur but should not alone determine the
diagnosis. An apparently simple cyst without specific signs can remain ambiguous.
These are CE-oriented cues, not a complete criterion set for alveolar
echinococcosis (AE). Do not rule out all hepatic echinococcosis merely because
these CE cues are absent. Do not assign a WHO stage.
""" + "\n" + OUTPUT_FORMAT)
