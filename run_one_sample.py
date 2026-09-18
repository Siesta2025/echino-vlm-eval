import json
from getpass import getpass
from pathlib import Path
from time import perf_counter

from openai import APIError

from dataset import DataLoader
from evaluate import Evaluator
from predictions import parse_prediction
from prompts import RawPrompt
from providers import OpenAIProvider, GLMProvider, image_to_data_url


ROOT = Path(__file__).resolve().parent

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


def main(dry_run: bool = True):
    # 1. Dataset provides the whole dev batch.
    data = DataLoader("dev", ROOT / "data/manifests/dev.jsonl").load_data()
    print(f"{len(data)} dev cases; at most {2 * len(data)} API calls.")
    if dry_run:
        print("Dry run: no API calls. Call main(dry_run=False) to enable them.")
        return

    # 2. Two models use the same batch and prompt.
    api_key = getpass("Enter your OpenRouter API key: ")
    host = "https://openrouter.ai/api/v1"
    openai_model = OpenAIProvider("openrouter", api_key, "openai/gpt-6-astra", host)
    glm_model = GLMProvider("openrouter", api_key, "z-ai/glm-5.3-flash", host)
    output = ROOT / "outputs/dev-batch"
    output.mkdir(parents=True, exist_ok=False)
    run_batch(data, openai_model, RawPrompt, output / "gpt.jsonl")
    run_batch(data, glm_model, RawPrompt, output / "glm.jsonl")

    # 3. Each evaluator compares its model's complete batch against the same gold.
    gpt_evaluator = Evaluator("gpt", data, output / "gpt.jsonl", output / "gpt-evaluation.jsonl")
    glm_evaluator = Evaluator("glm", data, output / "glm.jsonl", output / "glm-evaluation.jsonl")
    print("GPT:", gpt_evaluator.evaluate())
    print("GLM:", glm_evaluator.evaluate())


if __name__ == "__main__":
    main()
