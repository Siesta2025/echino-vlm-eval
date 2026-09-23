# Experiment configurations

Each JSON file declares the differences for one frozen experiment. `run.py`
executes one `model × condition` pair at a time; experiment logic does not live
in these files.

## Fields

- `name`: experiment identifier saved with the outputs.
- `data_path`: target-case JSONL manifest.
- `output_path`: default result directory.
- `models`: model aliases enabled for the experiment.
- `conditions`: named inference conditions.
- `prompt`: optional prompt registry key when it differs from the condition name.
- `skill`: optional versioned `SKILL.md` injected before the task prompt.
- `harness`: optional workflow; defaults to `single_pass`. `two_pass` performs a
  fixed visual observation followed by diagnosis. It is not an autonomous agent.
- `demonstrations`: optional few-shot JSONL manifest.
- `expected_demonstration_labels`: validates the frozen demonstration order.
- `artifacts`: optional auxiliary files whose hashes are recorded.

`full_v1.json` documents the reported experiment but requires the non-distributed
study data. For a new dataset, copy `example.json`, update `data_path`, and create
a manifest following the schema in the root README.

Validate a configuration without making API calls:

```bash
python run.py \
  --experiment experiments/example.json \
  --model gpt \
  --condition evidence \
  --dry-run
```

Remove `--dry-run` to execute. Use `--resume` only for an interrupted run whose
saved configuration and inference prefix still match. For an already completed
condition, `--retry-failures` retries only `api_error` and `incomplete` records,
preserving previous attempts. `--retry-timeout` controls that retry allowance.
