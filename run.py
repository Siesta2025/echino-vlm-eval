import json
import os
from dotenv import load_dotenv
from pathlib import Path
from time import perf_counter

from openai import APIError

from dataset import DataLoader
from evaluate import Evaluator
from predictions import parse_prediction
from prompts import RawPrompt, StructuredPrompt, WHOGuidedPrompt
from providers import OpenAIProvider, GLMProvider, image_to_data_url


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
config = {
    "batch_name": "dev-batch",
    "data_path": "data/manifests/dev.jsonl",
    "output_path": "outputs/dev-batch",
    "prompts": {
        "raw": RawPrompt,
        "structured": StructuredPrompt,
        "who_guided": WHOGuidedPrompt,
    },
    "run_models": ["gpt"],  # Add GLM after its official endpoint is verified.
    "models": {
        "gpt": {
            "name": "modelbest",
            "api_key_env": "GPT_API_KEY",
            "model": "gpt-6-astra", "base_url": "https://llm-center.modelbest.co/v1",
            "settings": {"stream": False},
        },
        "glm": {
            "name": "openrouter",
            "api_key_env": "GLM_API_KEY",
            "model": "z-ai/glm-5.3-flash", "base_url": "https://openrouter.ai/api/v1",
            "settings": {"reasoning_effort": "low", "max_tokens": 2048},
        },
    },
}

def run_batch(data: list[dict], model, prompt, result_path: Path):
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as f:
        for item in data:
            image_data_url = image_to_data_url(ROOT / item["image_path"])
            start = perf_counter()
            try:
                result = model.infer(str(prompt), image_data_url)
                if result.get("refusal"):
                    parsed = {"status": "refusal", "prediction": None, "error": "Model refused"}
                elif result["api_status"] == "incomplete":
                    parsed = {"status": "incomplete", "prediction": None, "error": "Output incomplete"}
                elif result["api_status"] != "completed":
                    parsed = {"status": "api_error", "prediction": None, "error": "API did not complete"}
                else:
                    parsed = parse_prediction(result["raw_text"])
                result.update(parsed)
            except APIError as exc:
                result = {
                    "status": "api_error", "prediction": None, "raw_text": None,
                    "error": type(exc).__name__, "api_status": None, "usage": None,
                    "elapsed_seconds": perf_counter() - start,
                    "requested_model": model.model_name, "host": model.base_url,
                    "settings": model.settings,
                }
            result["case_id"] = item["case_id"]
            result["attempts"] = 1
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    # 1. Dataset provides the whole dev batch.
    data = DataLoader(config["batch_name"] + "_dataloader", ROOT / config["data_path"]).load_data()

    # 2. Initialize selected models, using each service's own credentials.
    output = ROOT / config["output_path"]
    output.mkdir(parents=True, exist_ok=False)

    models = {}
    adapters = {"gpt": OpenAIProvider, "glm": GLMProvider}
    for name in config["run_models"]:
        cfg = config["models"][name]
        models[name] = adapters[name](
            cfg["name"], os.environ[cfg["api_key_env"]], cfg["model"],
            cfg["base_url"], settings=cfg["settings"],
        )

    for prompt_name, prompt in config["prompts"].items():
        for name, model in models.items():
            saved_config = {
                "batch_name": config["batch_name"],
                "data_path": config["data_path"],
                "prompt_name": prompt_name,
                "prompt": str(prompt),
                "model": model.model_name,
                "host": model.base_url,
                "settings": model.settings,
            }
            with (output / f"{name}-{prompt_name}.config.json").open("x", encoding="utf-8") as f:
                json.dump(saved_config, f, ensure_ascii=False, indent=2)
            run_batch(data, model, prompt, output / f"{name}-{prompt_name}-inference.jsonl")
    
    # 3. Each evaluator compares its model's complete batch against the same gold.
    summary = {}
    for name in models:
        summary[name] = {}
        for prompt_name in config["prompts"]:
            evaluator = Evaluator(
                name, data, output / f"{name}-{prompt_name}-inference.jsonl",
                output / f"{name}-{prompt_name}-evaluation.jsonl",
            )
            summary[name][prompt_name] = evaluator.evaluate()
    print(summary)
    with (output / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
