# Evaluating Vision-Language Models for Hepatic Echinococcosis on Ultrasound

This repository is a compact evaluation framework for testing whether general-purpose
vision-language models (VLMs) can recognize hepatic echinococcosis from a single
ultrasound image.

We compare two models across prompt-only baselines, evidence elicitation, multimodal
few-shot prompting, domain-skill injection, and a fixed two-pass reasoning harness.
The main result is deliberately modest: additional knowledge and inference steps do
not consistently improve performance. They can alter the sensitivity–specificity
trade-off while increasing latency, formatting failures, and incomplete responses.

> **Research use only.** This is a small exploratory study, not a clinical diagnostic
> system or a medical-device evaluation.

## Abstract

We evaluated `gpt-6-astra` and `glm-5.3-flash` on 30 public educational liver
ultrasound cases: 8 hepatic echinococcosis cases and 22 non-echinococcal lesions,
including 16 deliberately selected hard negatives. Seven controlled conditions were
tested for each model. The best conditions correctly classified 22/30 cases: GPT with
evidence elicitation or fixed few-shot demonstrations, and GLM with a structured
prompt. Explicit WHO-oriented cues increased false positives. A domain skill improved
GPT's accuracy among completed classifications but did not increase the total number
of correct cases. A two-pass observe-then-diagnose workflow improved rejection of some
hard negatives, but reduced GPT sensitivity from 8/8 to 4/8 and substantially reduced
GLM completion rate. These findings suggest that medical VLM evaluation should report
coverage, failure modes, and inference cost alongside diagnostic accuracy.

## Experimental design

### Task

Each request contains one ultrasound image and no diagnosis, caption, source title,
patient history, or gold label. The model returns one of:

- `positive`: visible findings favor hepatic echinococcosis;
- `negative`: visible findings favor another interpretation;
- `indeterminate`: the image does not support a meaningful choice.

Failures such as API errors, incomplete generations, refusals, and invalid JSON are
kept separate from medical negative predictions.

### Models

- `gpt-6-astra`, accessed through an OpenAI-compatible hosted endpoint;
- `glm-5.3-flash`, accessed through the official Zhipu API.

### Conditions

| Condition | Intervention | Model calls per case |
| --- | --- | ---: |
| `direct` | Minimal image-level classification prompt | 1 |
| `structured` | Adds morphology and differential-diagnosis checks | 1 |
| `who_guided` | Adds concise WHO-IWGE cystic-echinococcosis cues | 1 |
| `evidence` | Requests visible findings, an alternative, and a prediction | 1 |
| `fewshot_evidence` | Adds four fixed multimodal demonstrations | 1 |
| `skill_evidence` | Injects a versioned domain skill | 1 |
| `skill_two_pass` | First observes morphology, then diagnoses with the same skill | 2 |

The two-pass condition is a deterministic harness, not an autonomous agent: it has no
planning loop, dynamic tool selection, or access to external clinical information.

## Results

`Accuracy` is calculated only over classified cases. `Correct/all`, `TP`, and `TN`
retain the full denominators and therefore expose selective failures or abstentions.

| Model | Condition | Coverage | Accuracy | Correct/all | TP / 8 | TN / 22 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT | `direct` | 30/30 | 70.0% | 21/30 | 8 | 13 |
| GPT | `structured` | 30/30 | 66.7% | 20/30 | 8 | 12 |
| GPT | `who_guided` | 30/30 | 60.0% | 18/30 | 8 | 10 |
| GPT | `evidence` | 30/30 | 73.3% | **22/30** | 7 | 15 |
| GPT | `fewshot_evidence` | 30/30 | 73.3% | **22/30** | 8 | 14 |
| GPT | `skill_evidence` | 27/30 | **77.8%** | 21/30 | 7 | 14 |
| GPT | `skill_two_pass` | 27/30 | 70.4% | 19/30 | 4 | 15 |
| GLM | `direct` | 30/30 | 60.0% | 18/30 | 7 | 11 |
| GLM | `structured` | 30/30 | **73.3%** | **22/30** | 6 | 16 |
| GLM | `who_guided` | 27/30 | 37.0% | 10/30 | 7 | 3 |
| GLM | `evidence` | 26/30 | 65.4% | 17/30 | 6 | 11 |
| GLM | `fewshot_evidence` | 27/30 | 66.7% | 18/30 | 6 | 12 |
| GLM | `skill_evidence` | 19/30 | 52.6% | 10/30 | 4 | 6 |
| GLM | `skill_two_pass` | 16/30 | 62.5% | 10/30 | 4 | 6 |

The aggregate, machine-readable results are available in
[`results/summary.csv`](results/summary.csv). Images and raw model responses are
not distributed.

### Main observations

1. **Simple baselines remained competitive.** GPT direct classified every positive
   case correctly, although it produced nine false positives. GLM structured reached
   the best GLM result while retaining full coverage.

2. **Disease-specific cues caused anchoring.** WHO-guided prompting reduced true
   negatives from 13 to 10 for GPT and from 11 to 3 for GLM. Generic cystic, septated,
   calcified, or heterogeneous findings were frequently overinterpreted as disease-
   specific signs.

3. **Evidence elicitation was model-dependent.** GPT evidence improved specificity at
   the cost of one false negative. GLM evidence introduced additional schema failures
   and did not outperform the simpler structured prompt.

4. **Few-shot gains did not transfer consistently.** GPT few-shot performed well on
   the original cases but did not improve the hard-negative subset. Four examples were
   insufficient to represent the differential-diagnosis distribution.

5. **Skill-conditioned accuracy was affected by selective coverage.** GPT skill had
   the highest classified-only accuracy, but two API failures and one abstention left
   it with fewer total correct cases than GPT evidence. GLM skill produced many
   incomplete or unparsable responses.

6. **Two-pass reasoning mainly shifted the decision threshold.** GPT correctly rejected
   more hard negatives but missed four of eight positives. For GLM, all observation
   steps completed, while the diagnosis step produced 13 incomplete responses and one
   parse error.

The most persistent false positives were biliary cystadenoma, chronic hepatic
brucelloma, mesenchymal hamartoma, and complicated liver abscesses. Their complex
cystic, inflammatory, calcified, or necrotic appearances were repeatedly mapped to
echinococcosis-related features.

## Evaluation architecture

```text
experiment config
    ├── prompt
    ├── optional domain skill
    ├── optional demonstrations
    └── inference harness
             │
ultrasound ──┼──> provider adapter ──> raw response
             │                            │
             └────────────────────────────┤
                                          v
                              parser + status taxonomy
                                          │
                              case-level inference JSONL
                                          │
                                      evaluator
                                          │
                              evaluation JSONL + summary
```

The implementation follows four practical rules:

- dataset truth is never included in inference requests;
- provider-specific request syntax is isolated behind a common interface;
- raw outputs and operational failures are persisted before evaluation;
- experiments are configuration-driven and identified by content hashes.

## Repository structure

```text
.
├── dataset.py        # JSONL manifest loading
├── prompts.py        # versioned prompt registry and skill composition
├── providers.py      # OpenAI-compatible and Zhipu API adapters
├── predictions.py    # strict output parsing
├── harness.py        # fixed observe -> diagnose workflow
├── evaluate.py       # coverage, status, and confusion-matrix evaluation
├── run.py            # experiment runner, resume, and bounded retry logic
├── experiments/      # declarative experiment configurations
├── skills/           # versioned domain instructions
├── scripts/          # dataset-specific preparation utilities
├── results/           # aggregate, de-identified experiment metrics
└── tests/             # data-independent offline tests
```

The image data and model outputs are intentionally not distributed in this repository.

## Quick start

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Then replace the placeholders in `.env` with your own credentials.

Never commit API keys, patient identifiers, private clinical images, or Base64 image
payloads.

### Expected manifest

Prepare a JSONL file in which each line contains:

```json
{
  "case_id": "case-001",
  "group_id": "patient-or-source-001",
  "image_path": "path/to/image.png",
  "gold_label": "positive"
}
```

`gold_label` is used only by the evaluator. The provider receives the prompt and image,
not the manifest record.

### Test and run

```bash
pytest -q

python run.py \
  --experiment experiments/example.json \
  --model gpt \
  --condition evidence \
  --dry-run
```

Remove `--dry-run` to call the configured API. One invocation runs exactly one
model–condition pair. Results are written as a frozen configuration, case-level raw
inference records, evaluation records, and a summary.

```bash
python run.py \
  --experiment experiments/example.json \
  --model gpt \
  --condition evidence
```

Use `--resume` only for an interrupted, configuration-matching run. The runner can also
retry only infrastructure failures while preserving previous attempts:

```bash
python run.py \
  --experiment experiments/example.json \
  --model gpt \
  --condition evidence \
  --retry-failures \
  --retry-timeout 90
```

Successful predictions and parse errors are not resampled by this command.

## Limitations

- The dataset is very small, with only eight positive cases.
- Cases are public educational figures and may overlap with model pretraining data.
- The hard-negative subset was deliberately selected and does not represent clinical
  prevalence.
- Labels originate from source-case diagnoses and were not independently adjudicated
  by multiple clinicians for this study.
- Earlier exploratory outputs informed prompt development, so this is not a blind
  held-out evaluation.
- A single static image omits dynamic scanning, multiple views, and clinical context.
- One inference per condition is insufficient to estimate sampling variance.

The results should therefore be interpreted as an engineering and evaluation case
study, not as evidence of clinical readiness or a definitive ranking of the models.

## Related work

- Yang et al., [*Ultrasound identification of hepatic echinococcosis using a deep
  convolutional neural network model in China*](https://pubmed.ncbi.nlm.nih.gov/37507196/),
  *The Lancet Digital Health*, 2023.
- WHO, [*WHO guidelines for the treatment of patients with cystic echinococcosis*](https://www.who.int/publications/i/item/9789240110472),
  2025.
- [U2-BENCH: Benchmarking Large Vision-Language Models on Ultrasound Understanding](https://arxiv.org/abs/2505.17779).
- [MedBLINK: Probing Basic Perception in Multimodal Language Models for Medicine](https://arxiv.org/abs/2508.02951).
- [VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models](https://arxiv.org/abs/2407.11691).

## Future work

The most important next step is a genuinely held-out, patient-level dataset with
authorized use and expert-adjudicated labels. With such data, useful extensions would
include repeated trials and paired confidence intervals, expert evaluation of evidence
fidelity, controlled crop/zoom tools, CE/AE subtyping, WHO staging, and comparison with
ultrasound-specific encoders, linear probes, or LoRA adaptation.

## License

The source code is released under the [MIT License](LICENSE). Third-party datasets and
images are not included and remain subject to their original licenses and terms of use.
