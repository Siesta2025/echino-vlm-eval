import base64
from pathlib import Path
from PIL import Image
from time import perf_counter
from openai import OpenAI
from getpass import getpass
import json

import pytest
from predictions import parse_prediction


def image_to_data_url(image_path: Path) -> str:
    mime_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
    }
    with Image.open(image_path) as img:
        if img.format not in mime_types:
            raise ValueError(f"Unsupported image format: {img.format}")
        mime_type = mime_types[img.format]
    
    image_bytes = image_path.read_bytes()
    base64_bytes = base64.b64encode(image_bytes)
    image_base64 = base64_bytes.decode("ascii")
    return f"data:{mime_type};base64,{image_base64}"

def build_request(prompt: str, image_data_url: str, model: str) -> dict:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high",
                    }
                ]
            }
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 2048,
        "store": False,
    }

def main(dry_run: bool = True):
    if dry_run:
        print("Dry run mode: no API calls will be made.")
        
    # 1. load image 
    image_path = Path("./data/raw/medthinkvqa/images/case11668/_k57_T9b.jpg")
    image_data_url = image_to_data_url(image_path)
    header, payload = image_data_url.split(",", 1)
    assert base64.b64decode(payload, validate=True) == image_path.read_bytes()
    print(f"Image encoding verified: {header}; {image_path.stat().st_size} bytes")

    # 2. construct input
    prompt = """
    For this research evaluation, classify the provided liver ultrasound
    image for hepatic echinococcosis.

    Choose exactly one prediction:
    - positive: the image supports hepatic echinococcosis.
    - negative: the image does not support hepatic echinococcosis.
    - indeterminate: the image is insufficient or ambiguous.

    Return only a JSON object with exactly one field, "prediction".
    Its value must be "positive", "negative", or "indeterminate".
    Do not include Markdown fences or additional text.
    """

    request = build_request(prompt, image_data_url, model="openai/gpt-6-astra")
    print(f"Request constructed: model={request['model']}; image detail=high")

    # 3. feed into model API
    if not dry_run:
        api_key = getpass("Enter your OpenRouter API key: ")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            timeout=30,
            max_retries=0,
            api_key=api_key
        )

        start = perf_counter()

        response = client.responses.create(**request)
        elapsed = perf_counter() - start
        result = parse_prediction(response.output_text)

        # 4. grab the output and display it
        print(response.output_text)
        print(result)
        print(response.usage)
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print("Status:", response.status)
        print("Incomplete details", response.incomplete_details)

# Offline tests use synthetic answers and never call a model API.
@pytest.mark.parametrize("label", ["positive", "negative", "indeterminate"])
def test_parse_valid_labels(label):
    assert parse_prediction(json.dumps({"prediction": label})) == {
        "status": "ok", "prediction": label, "error": None,
    }


@pytest.mark.parametrize("raw_text", [
    "not JSON", "[]", "{}",
    '{"prediction": ["positive"]}',
    '{"prediction": "banana"}',
])
def test_parse_invalid_outputs(raw_text):
    result = parse_prediction(raw_text)
    assert result["status"] == "parse_error"
    assert result["prediction"] is None


@pytest.mark.parametrize("image_data_url", [None, "data:image/png;base64,c3ludGhldGlj"])
def test_provider_infer(monkeypatch, image_data_url):
    from types import SimpleNamespace
    import providers

    requests = []

    def fake_create(**request):
        requests.append(request)
        return SimpleNamespace(
            output_text='{"prediction": "indeterminate"}',
            usage=None, status="completed", incomplete_details=None,
            error=None, id="mock-response", model="mock-model", output=[],
        )

    monkeypatch.setattr(
        providers, "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=fake_create)),
    )
    provider = providers.OpenAIProvider(
        name="mock", api_key="mock-key", model_name="mock-model",
        base_url="https://example.invalid/v1",
    )
    result = provider.infer("Synthetic test prompt", image_data_url)
    content = requests[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Synthetic test prompt"}
    assert len(content) == (1 if image_data_url is None else 2)
    if image_data_url is not None:
        assert content[1]["image_url"] == image_data_url
    assert result["raw_text"] == '{"prediction": "indeterminate"}'
    assert result["api_status"] == "completed"
    assert result["elapsed_seconds"] >= 0
    assert result["usage"] is None


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_glm_provider_infer(monkeypatch, finish_reason):
    from openai.types.chat import ChatCompletion
    from types import SimpleNamespace
    import providers

    requests = []

    def fake_create(**request):
        requests.append(request)
        return ChatCompletion(
            id="mock-glm", created=0, model="z-ai/glm-5.3-flash",
            object="chat.completion",
            choices=[{
                "index": 0, "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": '{"prediction":"positive"}'},
            }],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    monkeypatch.setattr(
        providers, "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
        ),
    )
    provider = providers.GLMProvider(
        name="openrouter", api_key="mock-key", model_name="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
    )
    image = "data:image/png;base64,c3ludGhldGlj"
    result = provider.infer("Synthetic prompt", image)
    assert requests[0]["messages"][0]["content"][1] == {
        "type": "image_url", "image_url": {"url": image},
    }
    assert requests[0]["reasoning_effort"] == "low"
    assert result["raw_text"] == '{"prediction":"positive"}'
    assert result["usage"]["total_tokens"] == 15
    assert result["api_status"] == ("completed" if finish_reason == "stop" else "incomplete")


def test_prompt_output_contract():
    from prompts import RawPrompt, StructuredPrompt, WHOGuidedPrompt, OUTPUT_FORMAT

    for prompt in (RawPrompt, StructuredPrompt, WHOGuidedPrompt):
        assert str(prompt).endswith(OUTPUT_FORMAT)
        assert "hepatic echinococcosis" in str(prompt)


def test_jsonl_loading_and_saving(tmp_path):
    from dataset import DataLoader
    from providers import Provider

    path = tmp_path / "results" / "mock.jsonl"
    provider = Provider("mock", "mock-key", "mock-model", "https://example.invalid")
    records = [
        {"case_id": "a", "raw_text": "合成答案", "usage": {"total_tokens": 5}},
        {"case_id": "b", "status": "api_error", "prediction": None},
    ]
    for record in records:
        provider.save(record, path)
    assert DataLoader("mock", path).load_data() == records
    assert "合成答案" in path.read_text(encoding="utf-8")


def test_evaluate_by_id(tmp_path):
    from evaluate import Evaluator

    samples = [
        {"case_id": str(i), "gold_label": "positive" if i % 2 == 0 else "negative"}
        for i in range(7)
    ]
    predictions = ["positive", "negative", "negative", "positive", "indeterminate", None]
    records = [
        {"case_id": str(i), "prediction": prediction,
         "status": "ok" if prediction is not None else "api_error"}
        for i, prediction in enumerate(predictions)
    ]
    input_path = tmp_path / "inference.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(r) for r in reversed(records)) + "\n\n", encoding="utf-8",
    )
    output_path = tmp_path / "evaluation.jsonl"
    evaluator = Evaluator("mock", samples, input_path, output_path)
    summary = evaluator.evaluate()
    assert summary["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert summary["n_abstained"] == summary["n_missing"] == 1
    assert summary["status_counts"]["api_error"] == 1
    assert summary["coverage"] == 4 / 7
    assert summary["accuracy_on_classified"] == 0.5
    assert summary["correct_over_all"] == 2 / 7
    with pytest.raises(FileExistsError):
        evaluator.evaluate()


@pytest.mark.parametrize("case_ids", [["a", "a"], ["unknown"]])
def test_evaluate_rejects_bad_ids(tmp_path, case_ids):
    from evaluate import Evaluator

    input_path = tmp_path / "inference.jsonl"
    input_path.write_text("".join(
        json.dumps({"case_id": case_id, "status": "ok", "prediction": "positive"}) + "\n"
        for case_id in case_ids
    ), encoding="utf-8")
    output_path = tmp_path / "evaluation.jsonl"
    evaluator = Evaluator("mock", [{"case_id": "a", "gold_label": "positive"}], input_path, output_path)
    with pytest.raises(ValueError):
        evaluator.evaluate()
    assert not output_path.exists()


def test_two_model_batches(tmp_path, monkeypatch):
    import run_one_sample as runner
    from dataset import DataLoader
    from evaluate import Evaluator

    data = DataLoader("dev", runner.ROOT / "data/manifests/dev.jsonl").load_data()
    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")

    class MockModel:
        def infer(self, prompt, image):
            assert isinstance(prompt, str)
            assert image == "synthetic-image"
            return {
                "raw_text": '{"prediction":"indeterminate"}',
                "api_status": "completed", "source": "mock",
            }

    for name in ("gpt", "glm"):
        path = tmp_path / f"{name}.jsonl"
        runner.run_batch(data, MockModel(), runner.RawPrompt, path)
        records = DataLoader("mock", path).load_data()
        assert len(records) == len(data)
        assert [r["case_id"] for r in records] == [s["case_id"] for s in data]
        assert all("experiment_id" not in r and "gold_label" not in r for r in records)
        summary = Evaluator(name, data, path, tmp_path / f"{name}-eval.jsonl").evaluate()
        assert summary["n_abstained"] == len(data)
        assert "experiment_id" not in summary


@pytest.mark.parametrize("api_status,refusal,status", [
    ("incomplete", None, "incomplete"),
    ("failed", None, "api_error"),
    ("completed", "Synthetic refusal", "refusal"),
])
def test_batch_does_not_parse_failed_answers(tmp_path, monkeypatch, api_status, refusal, status):
    import run_one_sample as runner
    from dataset import DataLoader

    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")

    class MockModel:
        def infer(self, prompt, image):
            return {
                "raw_text": '{"prediction":"positive"}',
                "api_status": api_status, "refusal": refusal,
            }

    path = tmp_path / "mock.jsonl"
    runner.run_batch([{"case_id": "a", "image_path": "synthetic.png"}], MockModel(), "prompt", path)
    record = DataLoader("mock", path).load_data()[0]
    assert record["status"] == status
    assert record["prediction"] is None


if __name__ == "__main__":
    main(dry_run=True)
