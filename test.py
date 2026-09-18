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
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    }
                ]
            }
        ],
        "stream": False,
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

    request = build_request(prompt, image_data_url, model="gpt-6-astra")
    print(f"Request constructed: model={request['model']}; protocol=Chat Completions")

    # 3. feed into model API
    if not dry_run:
        api_key = getpass("Enter your ModelBest API key: ")
        client = OpenAI(
            base_url="https://llm-center.modelbest.co/v1",
            timeout=30,
            max_retries=0,
            api_key=api_key
        )

        start = perf_counter()

        response = client.chat.completions.create(**request)
        elapsed = perf_counter() - start
        choice = response.choices[0]
        if choice.message.refusal:
            result = {"status": "refusal", "prediction": None}
        elif choice.finish_reason != "stop":
            result = {"status": "incomplete", "prediction": None}
        else:
            result = parse_prediction(choice.message.content or "")

        # 4. grab the output and display it
        print(choice.message.content)
        print(result)
        print(response.usage)
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print("Finish reason:", choice.finish_reason)

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
            choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(
                    content='{"prediction": "indeterminate"}', refusal=None,
                ),
            )],
            usage=None, id="mock-response", model="mock-model",
        )

    monkeypatch.setattr(
        providers, "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
        ),
    )
    provider = providers.OpenAIProvider(
        name="mock", api_key="mock-key", model_name="mock-model",
        base_url="https://example.invalid/v1",
    )
    result = provider.infer("Synthetic test prompt", image_data_url)
    content = requests[0]["messages"][0]["content"]
    assert requests[0]["stream"] is False
    assert content[0] == {"type": "text", "text": "Synthetic test prompt"}
    assert len(content) == (1 if image_data_url is None else 2)
    if image_data_url is not None:
        assert content[1]["image_url"] == {"url": image_data_url}
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
    import run as runner
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
    import run as runner
    from dataset import DataLoader

    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")

    class MockModel:
        def infer(self, prompt, image):
            return {
                "raw_text": '{"prediction":"positive"}',
                "api_status": api_status, "refusal": refusal,
            }

    path = tmp_path / "mock.jsonl"
    runner.run_batch([{"case_id": "a", "image_path": "synthetic.png"}], MockModel(), runner.RawPrompt, path)
    record = DataLoader("mock", path).load_data()[0]
    assert record["status"] == status
    assert record["prediction"] is None


@pytest.mark.parametrize("enable_glm", [False, True])
def test_main_saves_config_and_summary(tmp_path, monkeypatch, enable_glm):
    import run as runner

    class MockProvider:
        def __init__(self, name, api_key, model_name, base_url, settings):
            assert name != "openrouter" or enable_glm
            self.model_name = model_name
            self.base_url = base_url
            self.settings = settings

        def infer(self, prompt, image):
            return {"api_status": "completed", "raw_text": '{"prediction":"indeterminate"}'}

    output = tmp_path / "batch"
    key_requests = []

    def fake_getpass(message):
        key_requests.append(message)
        return "synthetic-key"

    monkeypatch.setitem(runner.config, "output_path", str(output))
    monkeypatch.setitem(runner.config["models"]["glm"], "enabled", enable_glm)
    monkeypatch.setattr(runner, "getpass", fake_getpass)
    monkeypatch.setattr(runner, "OpenAIProvider", MockProvider)
    monkeypatch.setattr(runner, "GLMProvider", MockProvider)
    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")
    runner.main(dry_run=False)
    saved_config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert saved_config["prompt"] == "RawPrompt"
    assert len(saved_config["prompt_sha256"]) == 64
    expected_models = {"gpt", "glm"} if enable_glm else {"gpt"}
    assert set(saved_config["models"]) == set(summary) == expected_models
    assert len(key_requests) == len(expected_models)
    assert saved_config["models"]["gpt"]["model"] == "gpt-6-astra"
    assert all(result["n_abstained"] == 1 for result in summary.values())
    assert "synthetic-key" not in (output / "config.json").read_text()
    assert not (output / "prompt.txt").exists()


def test_vision_smoke(tmp_path, monkeypatch):
    import vision_smoke

    calls = []
    monkeypatch.setattr(vision_smoke, "ROOT", tmp_path)
    monkeypatch.setattr(vision_smoke, "getpass", lambda message: "synthetic-key")
    monkeypatch.setattr(vision_smoke, "make_test_image", lambda path: "123456")
    monkeypatch.setattr(vision_smoke, "image_to_data_url", lambda path: "synthetic-image")

    class MockProvider:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, prompt, image):
            assert "123456" not in prompt
            assert image == "synthetic-image"
            calls.append(prompt)
            return {"api_status": "completed", "raw_text": "123456", "usage": None}

    monkeypatch.setattr(vision_smoke, "OpenAIProvider", MockProvider)
    vision_smoke.main()
    assert not calls
    vision_smoke.main(live=True)
    result = json.loads((tmp_path / "outputs/vision-smoke.json").read_text())
    assert result["vision_check_passed"] is True
    assert len(calls) == 1
    assert "synthetic-key" not in json.dumps(result)
    with pytest.raises(FileExistsError):
        vision_smoke.main(live=True)
    assert len(calls) == 1


def test_vision_test_image(tmp_path):
    from vision_smoke import make_test_image

    path = tmp_path / "test.png"
    expected = make_test_image(path)
    assert len(expected) == 6 and expected.isdigit()
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (512, 256)


if __name__ == "__main__":
    main(dry_run=True)
