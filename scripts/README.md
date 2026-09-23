# Dataset preparation scripts

This directory contains one-off or dataset-specific preparation utilities. Model
inference and experiment orchestration remain in the root `run.py`; differences
between experiments belong in `experiments/*.json`.

## `prepare_extension_v1.py`

Reconstructs the manually reviewed 16-case hard-negative extension from a local
MedThinkVQA download.

Required local inputs:

- `data/raw/medthinkvqa/{train,test}.jsonl`
- `data/raw/medthinkvqa/images/`
- the frozen `SELECTED` and `EXCLUDED` lists in the script

Generated local artifacts:

- `data/manifests/extension-v1.jsonl`: neutral inference manifest
- `data/extension-v1/audit.jsonl`: provenance and label-evidence audit
- `data/extension-v1/images/`: unchanged copies of selected images

```bash
python scripts/prepare_extension_v1.py
```

The script makes no model API calls. If an existing generated file differs, it
stops instead of silently overwriting the reviewed artifact. Data are not
distributed with this repository; only the reproducible selection logic is.
