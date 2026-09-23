"""Run one model/condition pair from a frozen experiment configuration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from openai import APIError
from zai.core import ZaiError

from dataset import DataLoader
from evaluate import Evaluator
from harness import run_two_pass
from predictions import parse_prediction
from prompts import PROMPTS, add_skill
from providers import GLMProvider, OpenAIProvider, image_to_data_url


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MODEL_CONFIGS = {
    "gpt": {
        "name": "modelbest",
        "api_key_env": "GPT_API_KEY",
        "model": "gpt-6-astra",
        "base_url": "https://llm-center.modelbest.co/v1",
        "settings": {"stream": False},
    },
    "glm": {
        "name": "zhipu",
        "api_key_env": "GLM_API_KEY",
        "model": "glm-5.3-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "settings": {"stream": False, "max_tokens": 2048},
    },
}
ADAPTERS = {"gpt": OpenAIProvider, "glm": GLMProvider}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return DataLoader(path.stem, path).load_data()


def load_skill(path: Path) -> str:
    """Load the Markdown body; YAML metadata is for skill discovery, not the VLM."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        _, _, text = text.split("---\n", 2)
    return text.strip()


def parse_api_result(result: dict) -> dict:
    if result.get("status") == "api_error":
        return {
            "status": "api_error", "prediction": None,
            "error": result.get("error", "API did not complete"),
        }
    if result.get("refusal"):
        return {"status": "refusal", "prediction": None, "error": "Model refused"}
    if result["api_status"] == "incomplete":
        return {"status": "incomplete", "prediction": None, "error": "Output incomplete"}
    if result["api_status"] != "completed":
        return {"status": "api_error", "prediction": None, "error": "API did not complete"}
    return parse_prediction(result["raw_text"])


def call_api(model, prompt: str, image_urls: list[str]) -> dict:
    start = perf_counter()
    try:
        return model.infer(prompt, image_urls)
    except (APIError, ZaiError) as exc:
        return {
            "status": "api_error",
            "prediction": None,
            "raw_text": None,
            "error": type(exc).__name__,
            "api_status": None,
            "usage": None,
            "elapsed_seconds": perf_counter() - start,
            "requested_model": model.model_name,
            "host": model.base_url,
            "settings": model.settings,
            "transport_timeout_seconds": model.timeout_seconds,
        }


def infer_case(model, prompt: str, image_urls: list[str]) -> dict:
    result = call_api(model, prompt, image_urls)
    result.update(parse_api_result(result))
    return result


def build_saved_config(
    experiment: dict,
    model_name: str,
    condition_name: str,
    prompt: str,
    data_path: Path,
    demonstrations: list[dict],
) -> dict:
    model_config = MODEL_CONFIGS[model_name]
    return {
        "experiment": experiment["name"],
        "data_path": str(data_path.relative_to(ROOT)),
        "manifest_sha256": sha256(data_path.read_bytes()),
        "condition": condition_name,
        "prompt": prompt,
        "prompt_sha256": sha256(prompt.encode("utf-8")),
        "demonstrations": [
            {
                "case_id": item["case_id"],
                "label": item["gold_label"],
                "image_sha256": sha256((ROOT / item["image_path"]).read_bytes()),
            }
            for item in demonstrations
        ],
        "artifacts": {
            path: sha256((ROOT / path).read_bytes())
            for path in experiment.get("artifacts", [])
        },
        "model": model_config["model"],
        "host": model_config["base_url"],
        "settings": model_config["settings"],
    }


def run_condition(
    data: list[dict],
    demonstrations: list[dict],
    model,
    prompt: str,
    saved_config: dict,
    inference_path: Path,
    start_index: int = 0,
    append: bool = False,
    harness_name: str = "single_pass",
) -> None:
    demonstration_images = [
        image_to_data_url(ROOT / item["image_path"])
        for item in demonstrations
    ]
    mode = "a" if append else "x"
    with inference_path.open(mode, encoding="utf-8") as output:
        for item in data[start_index:]:
            image_path = ROOT / item["image_path"]
            image_urls = demonstration_images + [image_to_data_url(image_path)]
            if harness_name == "single_pass":
                result = infer_case(model, prompt, image_urls)
            elif harness_name == "two_pass":
                result = run_two_pass(
                    model, prompt, image_urls, call_api, parse_api_result,
                )
            else:
                raise ValueError(f"Unknown harness: {harness_name}")
            result.update({
                "case_id": item["case_id"],
                "image_sha256": sha256(image_path.read_bytes()),
                "prompt_sha256": saved_config["prompt_sha256"],
                "demonstration_case_ids": [d["case_id"] for d in demonstrations],
                "attempts": 1,
            })
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"{item['case_id']}: {result['status']} / {result['prediction']}",
                flush=True,
            )
            if result.get("error") in ("APIConnectionError", "ApiConnectionError"):
                raise RuntimeError("Connection failed; resume this condition later")


def retry_failed_records(
    data: list[dict],
    demonstrations: list[dict],
    model,
    prompt: str,
    saved_config: dict,
    inference_path: Path,
    harness_name: str,
) -> int:
    """Retry infrastructure failures while preserving every previous attempt."""
    records = load_jsonl(inference_path)
    if len(records) != len(data):
        raise ValueError("Retry requires a complete inference file")

    demonstration_images = [
        image_to_data_url(ROOT / item["image_path"])
        for item in demonstrations
    ]
    retry_statuses = {"api_error", "incomplete"}
    retried = 0
    updated_records = []

    for item, old in zip(data, records):
        if old["case_id"] != item["case_id"]:
            raise ValueError(f"Saved record does not match: {item['case_id']}")
        if old["status"] not in retry_statuses:
            updated_records.append(old)
            continue

        image_path = ROOT / item["image_path"]
        image_urls = demonstration_images + [image_to_data_url(image_path)]
        if harness_name == "single_pass":
            result = infer_case(model, prompt, image_urls)
        else:
            result = run_two_pass(
                model, prompt, image_urls, call_api, parse_api_result,
            )

        history = old.get("attempt_history", []) + [{
            key: old.get(key) for key in (
                "status", "prediction", "raw_text", "error", "api_status",
                "usage", "elapsed_seconds", "harness_trace",
                "transport_timeout_seconds",
            )
        }]
        result.update({
            "case_id": item["case_id"],
            "image_sha256": old["image_sha256"],
            "prompt_sha256": saved_config["prompt_sha256"],
            "demonstration_case_ids": old["demonstration_case_ids"],
            "attempts": old.get("attempts", 1) + 1,
            "attempt_history": history,
        })
        updated_records.append(result)
        retried += 1
        print(
            f"{item['case_id']}: retry -> {result['status']} / {result['prediction']}",
            flush=True,
        )

    temporary_path = inference_path.with_suffix(".retry.jsonl")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record in updated_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(inference_path)
    return retried


def validate_resume(
    data: list[dict],
    saved_config: dict,
    config_path: Path,
    inference_path: Path,
) -> int:
    if not config_path.exists() or not inference_path.exists():
        raise FileNotFoundError("No saved condition to resume")
    if json.loads(config_path.read_text(encoding="utf-8")) != saved_config:
        raise ValueError("Saved config does not match the requested condition")

    previous = load_jsonl(inference_path)
    if len(previous) > len(data):
        raise ValueError("Saved inference contains more records than the dataset")
    for item, record in zip(data, previous):
        image_path = ROOT / item["image_path"]
        if (
            record["case_id"] != item["case_id"]
            or record["image_sha256"] != sha256(image_path.read_bytes())
            or record["prompt_sha256"] != saved_config["prompt_sha256"]
            or record["requested_model"] != saved_config["model"]
        ):
            raise ValueError(f"Saved record does not match: {item['case_id']}")
    return len(previous)


def run_experiment(
    experiment_path: Path,
    model_name: str,
    condition_name: str,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    retry_failures: bool = False,
    retry_timeout: float = 30,
    transport_timeout: float = 30,
) -> dict | None:
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if model_name not in experiment["models"]:
        raise ValueError(f"Model not enabled for this experiment: {model_name}")
    if condition_name not in experiment["conditions"]:
        raise ValueError(f"Condition not enabled for this experiment: {condition_name}")
    condition = experiment["conditions"][condition_name]
    prompt_name = condition.get("prompt", condition_name)
    if prompt_name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {prompt_name}")
    data_path = ROOT / experiment["data_path"]
    data = load_jsonl(data_path)
    demonstration_path = condition.get("demonstrations")
    demonstrations = load_jsonl(ROOT / demonstration_path) if demonstration_path else []
    expected_labels = condition.get("expected_demonstration_labels")
    if expected_labels is not None:
        labels = [item["gold_label"] for item in demonstrations]
        if labels != expected_labels:
            raise ValueError("Demonstration labels do not match the frozen order")
    if {item["group_id"] for item in data} & {item["group_id"] for item in demonstrations}:
        raise ValueError("Target and demonstration groups overlap")

    prompt = PROMPTS[prompt_name]
    skill_relative_path = condition.get("skill")
    skill_path = ROOT / skill_relative_path if skill_relative_path else None
    if skill_path:
        prompt = add_skill(prompt, load_skill(skill_path))
    prompt = str(prompt)
    saved_config = build_saved_config(
        experiment, model_name, condition_name, prompt, data_path, demonstrations,
    )
    saved_config.update({
        "prompt_name": prompt_name,
        "skill_path": skill_relative_path,
        "skill_sha256": sha256(skill_path.read_bytes()) if skill_path else None,
        "harness": condition.get("harness", "single_pass"),
    })
    if transport_timeout != 30:
        saved_config["transport_timeout_seconds"] = transport_timeout
    destination = output_dir or ROOT / experiment["output_path"]
    stem = f"{model_name}-{condition_name}"
    config_path = destination / f"{stem}.config.json"
    inference_path = destination / f"{stem}-inference.jsonl"
    evaluation_path = destination / f"{stem}-evaluation.jsonl"
    summary_path = destination / f"{stem}-summary.json"

    if dry_run:
        print(
            f"{experiment['name']} / {stem}: {len(data)} targets, "
            f"{len(demonstrations)} demonstrations, no API calls"
        )
        return None
    if retry_failures:
        if config_path.exists():
            original_config = json.loads(config_path.read_text(encoding="utf-8"))
            if "transport_timeout_seconds" in original_config:
                saved_config["transport_timeout_seconds"] = original_config[
                    "transport_timeout_seconds"
                ]
        validate_resume(data, saved_config, config_path, inference_path)
        model_config = MODEL_CONFIGS[model_name]
        model = ADAPTERS[model_name](
            model_config["name"],
            os.environ[model_config["api_key_env"]],
            model_config["model"],
            model_config["base_url"],
            settings=model_config["settings"],
            timeout=retry_timeout,
        )
        retried = retry_failed_records(
            data, demonstrations, model, prompt, saved_config, inference_path,
            saved_config["harness"],
        )
        summary = Evaluator(
            model_name, data, inference_path, evaluation_path,
        ).evaluate(overwrite=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(
            f"{stem} retried {retried}: {summary['n_classified']}/{len(data)} classified"
        )
        return summary

    if evaluation_path.exists() or summary_path.exists():
        raise FileExistsError(f"Condition already evaluated: {stem}")

    destination.mkdir(parents=True, exist_ok=True)
    if resume:
        start_index = validate_resume(data, saved_config, config_path, inference_path)
        print(f"{stem}: continuing after {start_index}/{len(data)} records", flush=True)
    else:
        if config_path.exists() or inference_path.exists():
            raise FileExistsError(f"Condition already has outputs: {stem}")
        config_path.write_text(
            json.dumps(saved_config, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        start_index = 0

    model_config = MODEL_CONFIGS[model_name]
    model = ADAPTERS[model_name](
        model_config["name"],
        os.environ[model_config["api_key_env"]],
        model_config["model"],
        model_config["base_url"],
        settings=model_config["settings"],
        **({"timeout": transport_timeout} if transport_timeout != 30 else {}),
    )
    run_condition(
        data, demonstrations, model, prompt, saved_config, inference_path,
        start_index, append=resume, harness_name=saved_config["harness"],
    )
    summary = Evaluator(
        model_name, data, inference_path, evaluation_path,
    ).evaluate()
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"{stem} complete: {summary['n_classified']}/{len(data)} classified")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--retry-timeout", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    run_experiment(
        args.experiment,
        args.model,
        args.condition,
        args.output_dir,
        args.dry_run,
        args.resume,
        args.retry_failures,
        args.retry_timeout,
        args.timeout,
    )


if __name__ == "__main__":
    main()
