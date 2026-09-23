class Prompt:
    """Small value object passed to Provider.infer via str(prompt)."""

    def __init__(self, prompt: str):
        self.prompt = prompt

    def __str__(self):
        return self.prompt


def add_skill(prompt: Prompt, skill_text: str) -> Prompt:
    """Place reusable domain instructions before the task-specific prompt."""
    prefix = """Use the following versioned domain skill as reference guidance.
Do not assume a listed feature is present; judge the target image itself.

<domain_skill>
"""
    return Prompt(prefix + skill_text.strip() + "\n</domain_skill>\n\n" + str(prompt))

# V1 is the original pilot condition. Keep its text unchanged so old outputs
# remain reproducible.
TASK_V1 = """For research evaluation only, classify the provided liver ultrasound
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

RawPrompt = Prompt(TASK_V1 + "\n" + OUTPUT_FORMAT)

StructuredPrompt = Prompt(TASK_V1 + """
Before choosing a label, assess whether the image is interpretable, inspect the
lesion's boundary and internal structure, and weigh an echinococcal interpretation
against plausible alternatives. Do not invent features that are not visible.
If the available view cannot distinguish these interpretations, abstain.
""" + "\n" + OUTPUT_FORMAT)

WHOGuidedPrompt = Prompt(TASK_V1 + """
WHO-IWGE ultrasound descriptions for cystic echinococcosis (CE) include a double
wall, daughter cysts, and detached internal membranes. Degenerated heterogeneous
contents or a calcified wall may also occur but should not alone determine the
diagnosis. An apparently simple cyst without specific signs can remain ambiguous.
These are CE-oriented cues, not a complete criterion set for alveolar
echinococcosis (AE). Do not rule out all hepatic echinococcosis merely because
these CE cues are absent. Do not assign a WHO stage.
""" + "\n" + OUTPUT_FORMAT)

PROMPTS_V1 = {
    "raw": RawPrompt,
    "structured": StructuredPrompt,
    "who_guided": WHOGuidedPrompt,
}


# V2 was written after reviewing V1. It asks for a best-supported binary choice
# whenever the image is interpretable, so it is an exploratory revision rather
# than an independently validated improvement.
TASK_V2 = """For research evaluation only, classify the provided liver ultrasound
image for hepatic echinococcosis. Use visible image evidence only; ignore
diagnostic text overlays and do not invent clinical history.

Choose exactly one prediction:
- positive: the visible lesion favors hepatic echinococcosis.
- negative: the visible lesion favors a non-echinococcal interpretation.
- indeterminate: the image is unreadable, no relevant liver lesion can be
identified, or the view cannot support even a tentative comparison.

If a relevant lesion is interpretable, make the best-supported positive or
negative choice despite uncertainty. This is an image-level research judgment,
not a definitive patient diagnosis.
"""

RawPromptV2 = Prompt(TASK_V2 + "\n" + OUTPUT_FORMAT)

StructuredPromptV2 = Prompt(TASK_V2 + """
Compare lesion shape, boundary, internal architecture, and visible contents
with plausible non-echinococcal liver lesions. Choose the interpretation better
supported by the visible features; do not require diagnostic certainty.
""" + "\n" + OUTPUT_FORMAT)

WHOGuidedPromptV2 = Prompt(TASK_V2 + """
For cystic echinococcosis, daughter cysts, detached membranes, and a double
wall can support a positive interpretation. A simple cyst without distinctive
features is not automatically negative; compare it with other possibilities.
These CE-oriented cues do not fully describe alveolar echinococcosis. Do not
assign a WHO stage or rule out all echinococcosis solely because CE cues are
absent.
""" + "\n" + OUTPUT_FORMAT)

PROMPTS_V2 = {
    "raw_v2": RawPromptV2,
    "structured_v2": StructuredPromptV2,
    "who_guided_v2": WHOGuidedPromptV2,
}


# Prompt-only ablations run after V2. Both use the same evidence schema so the
# few-shot comparison changes demonstrations, not the requested output shape.
EVIDENCE_OUTPUT_FORMAT = """Return only a JSON object with exactly these fields:
- "findings": an array of 1 to 3 short phrases describing visible image features.
- "alternative": the most plausible non-echinococcal alternative, or null.
- "prediction": "positive", "negative", or "indeterminate".
Do not include Markdown fences, hidden reasoning, subtype, stage, or treatment advice.
"""

EVIDENCE_INSTRUCTION = """
Before predicting, identify only features visible in the target image. Consider
whether a non-echinococcal lesion better explains those findings. The reported
findings are concise evidence for audit, not a definitive clinical diagnosis.
"""

EvidencePromptV1 = Prompt(
    TASK_V2 + EVIDENCE_INSTRUCTION + "\n" + EVIDENCE_OUTPUT_FORMAT
)

FewShotEvidencePromptV1 = Prompt(TASK_V2 + """
You will receive five liver ultrasound images in this fixed order:
1. labeled demonstration: positive
2. labeled demonstration: negative
3. labeled demonstration: positive
4. labeled demonstration: negative
5. unlabeled target image to classify

Use the demonstrations only to calibrate the task and label space. Describe and
classify image 5 only. Do not copy a demonstration label based on superficial
similarity, and do not infer clinical history from any image.
""" + EVIDENCE_INSTRUCTION + "\n" + EVIDENCE_OUTPUT_FORMAT)

PROMPT_ABLATIONS_V1 = {
    "evidence_v1": EvidencePromptV1,
    "fewshot_evidence_v1": FewShotEvidencePromptV1,
}

# The runner resolves a condition name through this single registry. Versioned
# groups above remain useful when reading the experimental history.
PROMPTS = {
    **PROMPTS_V1,
    **PROMPTS_V2,
    **PROMPT_ABLATIONS_V1,
}
