---
name: hepatic-echinococcosis-ultrasound
description: Review liver ultrasound images for research-only hepatic echinococcosis recognition using visible evidence and explicit differential diagnosis. Do not use for clinical care or treatment decisions.
---

# Hepatic echinococcosis ultrasound review

Use this skill only to structure an image-level research judgment. A single
ultrasound frame may not contain the evidence used for the source diagnosis.

## Review workflow

1. Check whether the target image contains interpretable liver parenchyma and a
   relevant lesion. Use `indeterminate` if the image is unreadable or cannot
   support even a tentative comparison.
2. Describe only visible morphology: lesion boundary, cystic versus solid
   appearance, internal architecture, echogenic contents, wall, membranes,
   septations, calcification or shadowing, and posterior acoustic behavior.
3. Look for combinations that can support cystic echinococcosis, including a
   double-wall appearance, daughter cysts, or detached/floating internal
   membranes. Degenerated heterogeneous contents and a calcified wall may occur,
   but are less specific when isolated.
4. Compare the observed pattern with plausible alternatives. A smooth simple
   cyst, abscess, biliary cystic neoplasm, haemorrhagic cyst, and necrotic or
   cystic tumour can overlap with individual echinococcal features. Naming an
   alternative is not enough; weigh which interpretation better fits the visible
   combination.
5. The cystic echinococcosis framework does not fully describe alveolar
   echinococcosis. Do not exclude all hepatic echinococcosis solely because
   classic cystic signs are absent.
6. Return `positive` when the visible pattern favors hepatic echinococcosis,
   `negative` when a non-echinococcal interpretation is better supported, and
   `indeterminate` when the image cannot support that comparison.

## Boundaries

- Treat annotations, arrows and calipers only as localization aids. Ignore
  diagnostic text overlays.
- Do not invent exposure history, symptoms, serology, CT/MRI findings, pathology
  or additional ultrasound views.
- Do not assign CE stage, CE versus AE subtype, activity, prognosis or treatment.
- Do not assume that a listed sign is present; image evidence must dominate this
  reference knowledge.
- Follow the caller's output schema exactly and keep evidence concise.

Clinical basis: [WHO guidelines for the treatment of patients with cystic
echinococcosis (2025)](https://www.who.int/publications/i/item/9789240110472) and
the associated [WHO-IWGE ultrasound classification
table](https://www.ncbi.nlm.nih.gov/books/NBK616291/table/ch1.tab1/). WHO notes
that diagnosis is imaging-led and may require complementary evidence when
imaging is inconclusive; this skill is not a complete diagnostic guideline.
