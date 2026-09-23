import json
from pathlib import Path

import pytest
from PIL import Image

from predictions import parse_prediction


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def synthetic_project(tmp_path):
    image_path = tmp_path / "images/case.png"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (8, 8), color=128).save(image_path)

    records = [
        {
            "case_id": f"case-{index}",
            "group_id": f"group-{index}",
            "image_path": "images/case.png",
            "gold_label": label,
        }
        for index, label in enumerate(
            ["positive", "negative", "positive", "negative"], start=1,
        )
    ]
    manifest_path = tmp_path / "data/manifests/dev.jsonl"
    write_jsonl(manifest_path, records)

    skill_path = tmp_path / "skills/synthetic/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\nname: synthetic\ndescription: Synthetic test skill.\n---\n\n"
        "# Synthetic skill\n\n## Review workflow\n\nUse visible evidence.\n",
        encoding="utf-8",
    )
    return {"root": tmp_path, "records": records, "manifest": manifest_path}


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


@pytest.mark.parametrize("image_data_url,n_images", [
    (None, 0),
    ("data:image/png;base64,c3ludGhldGlj", 1),
    (["data:image/png;base64,YQ==", "data:image/png;base64,Yg=="], 2),
])
def test_provider_infer(monkeypatch, image_data_url, n_images):
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
    assert len(content) == 1 + n_images
    if image_data_url is not None:
        expected = [image_data_url] if isinstance(image_data_url, str) else image_data_url
        assert [part["image_url"]["url"] for part in content[1:]] == expected
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
        providers, "ZhipuAiClient",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
        ),
    )
    provider = providers.GLMProvider(
        name="zhipu", api_key="mock-key", model_name="glm-5.3-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )
    image = "data:image/png;base64,c3ludGhldGlj"
    result = provider.infer("Synthetic prompt", image)
    assert requests[0]["messages"][0]["content"][1] == {
        "type": "image_url", "image_url": {"url": image},
    }
    assert requests[0]["stream"] is False
    assert result["raw_text"] == '{"prediction":"positive"}'
    assert result["usage"]["total_tokens"] == 15
    assert result["api_status"] == ("completed" if finish_reason == "stop" else "incomplete")


def test_prompt_output_contract():
    from prompts import OUTPUT_FORMAT, PROMPTS_V1, PROMPTS_V2, PROMPT_ABLATIONS_V1

    for prompt in (*PROMPTS_V1.values(), *PROMPTS_V2.values()):
        assert str(prompt).endswith(OUTPUT_FORMAT)
        assert "hepatic echinococcosis" in str(prompt)
    for prompt in PROMPT_ABLATIONS_V1.values():
        assert '"findings"' in str(prompt)
        assert '"alternative"' in str(prompt)
        assert '"prediction"' in str(prompt)


def test_skill_is_loaded_without_frontmatter():
    from run import ROOT, load_skill

    text = load_skill(
        ROOT / "skills/hepatic-echinococcosis-ultrasound/SKILL.md"
    )
    assert not text.startswith("---")
    assert "Review workflow" in text
    assert "pilot-img" not in text


def test_skill_condition_is_injected_and_recorded(
    tmp_path, monkeypatch, synthetic_project,
):
    import run as runner

    monkeypatch.setattr(runner, "ROOT", synthetic_project["root"])

    class MockProvider:
        def __init__(self, name, api_key, model_name, base_url, settings):
            self.model_name = model_name
            self.base_url = base_url
            self.settings = settings

        def infer(self, prompt, images):
            assert "<domain_skill>" in prompt
            assert "Review workflow" in prompt
            return {"api_status": "completed", "raw_text": '{"prediction":"indeterminate"}'}

    experiment_path = tmp_path / "skill.json"
    experiment_path.write_text(json.dumps({
        "name": "skill-test",
        "data_path": "data/manifests/dev.jsonl",
        "output_path": "unused",
        "models": ["gpt"],
        "conditions": {
            "skill_evidence_v1": {
                "prompt": "evidence_v1",
                "skill": "skills/synthetic/SKILL.md",
            },
        },
    }))
    monkeypatch.setenv("GPT_API_KEY", "synthetic-key")
    monkeypatch.setitem(runner.ADAPTERS, "gpt", MockProvider)
    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")

    output = tmp_path / "output"
    runner.run_experiment(
        experiment_path, "gpt", "skill_evidence_v1", output_dir=output,
    )
    saved = json.loads((output / "gpt-skill_evidence_v1.config.json").read_text())
    assert saved["prompt_name"] == "evidence_v1"
    assert saved["skill_path"].endswith("SKILL.md")
    assert len(saved["skill_sha256"]) == 64


def test_two_pass_harness_records_both_steps():
    from harness import run_two_pass
    from run import parse_api_result

    responses = iter([
        {
            "raw_text": json.dumps({
                "image_quality": "adequate",
                "lesion_present": True,
                "findings": ["synthetic cystic lesion"],
            }),
            "api_status": "completed", "usage": {"total_tokens": 10},
            "elapsed_seconds": 0.1,
        },
        {
            "raw_text": '{"prediction":"negative"}',
            "api_status": "completed", "usage": {"total_tokens": 20},
            "elapsed_seconds": 0.2,
        },
    ])
    prompts = []

    def fake_call(model, prompt, images):
        prompts.append(prompt)
        return next(responses)

    result = run_two_pass(None, "final prompt", ["image"], fake_call, parse_api_result)
    assert result["status"] == "ok"
    assert result["prediction"] == "negative"
    assert result["usage"]["total_tokens"] == 30
    assert result["elapsed_seconds"] == pytest.approx(0.3)
    assert [step["step"] for step in result["harness_trace"]] == ["observe", "diagnose"]
    assert "synthetic cystic lesion" in prompts[1]


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
    evaluator = Evaluator(
        "mock", [{"case_id": "a", "gold_label": "positive"}],
        input_path, output_path,
    )
    with pytest.raises(ValueError):
        evaluator.evaluate()
    assert not output_path.exists()


def test_two_model_batches(tmp_path, monkeypatch, synthetic_project):
    import run as runner
    from dataset import DataLoader
    from evaluate import Evaluator

    monkeypatch.setattr(runner, "ROOT", synthetic_project["root"])
    data = synthetic_project["records"]
    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")

    class MockModel:
        def infer(self, prompt, image):
            assert isinstance(prompt, str)
            assert image == ["synthetic-image"]
            return {
                "raw_text": '{"prediction":"indeterminate"}',
                "api_status": "completed", "source": "mock",
            }

    for name in ("gpt", "glm"):
        path = tmp_path / f"{name}.jsonl"
        prompt = str(runner.PROMPTS["raw"])
        saved_config = {"prompt_sha256": runner.sha256(prompt.encode())}
        runner.run_condition(data, [], MockModel(), prompt, saved_config, path)
        records = DataLoader("mock", path).load_data()
        assert len(records) == len(data)
        assert [r["case_id"] for r in records] == [s["case_id"] for s in data]
        assert all("experiment_id" not in r and "gold_label" not in r for r in records)
        summary = Evaluator(name, data, path, tmp_path / f"{name}-eval.jsonl").evaluate()
        assert summary["n_abstained"] == len(data)
        assert "experiment_id" not in summary


def test_resume_can_append_to_empty_inference_file(
    tmp_path, monkeypatch, synthetic_project,
):
    import run as runner

    monkeypatch.setattr(runner, "ROOT", synthetic_project["root"])
    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")

    class MockModel:
        def infer(self, prompt, images):
            return {
                "raw_text": '{"prediction":"positive"}',
                "api_status": "completed",
            }

    path = tmp_path / "inference.jsonl"
    path.touch()
    data = [{"case_id": "a", "image_path": "images/case.png"}]
    runner.run_condition(
        data, [], MockModel(), "prompt", {"prompt_sha256": "hash"},
        path, append=True,
    )
    record = json.loads(path.read_text())
    assert record["case_id"] == "a"
    assert record["prediction"] == "positive"


@pytest.mark.parametrize("api_status,refusal,status", [
    ("incomplete", None, "incomplete"),
    ("failed", None, "api_error"),
    ("completed", "Synthetic refusal", "refusal"),
])
def test_batch_does_not_parse_failed_answers(tmp_path, monkeypatch, api_status, refusal, status):
    import run as runner

    class MockModel:
        def infer(self, prompt, image):
            return {
                "raw_text": '{"prediction":"positive"}',
                "api_status": api_status, "refusal": refusal,
            }

    result = runner.infer_case(MockModel(), "prompt", ["synthetic-image"])
    assert result["status"] == status
    assert result["prediction"] is None


def test_api_exception_type_is_not_overwritten():
    import run as runner

    parsed = runner.parse_api_result({
        "status": "api_error",
        "prediction": None,
        "error": "APITimeoutError",
        "api_status": None,
    })
    assert parsed["error"] == "APITimeoutError"


@pytest.mark.parametrize("enable_glm", [False, True])
def test_main_saves_config_and_summary(
    tmp_path, monkeypatch, enable_glm, synthetic_project,
):
    import run as runner

    monkeypatch.setattr(runner, "ROOT", synthetic_project["root"])

    class MockProvider:
        def __init__(self, name, api_key, model_name, base_url, settings):
            self.model_name = model_name
            self.base_url = base_url
            self.settings = settings

        def infer(self, prompt, image):
            return {"api_status": "completed", "raw_text": '{"prediction":"indeterminate"}'}

    output = tmp_path / "batch"
    experiment_path = tmp_path / "experiment.json"
    experiment_path.write_text(json.dumps({
        "name": "synthetic",
        "data_path": "data/manifests/dev.jsonl",
        "output_path": "unused",
        "models": ["gpt", "glm"],
        "conditions": {"raw": {}},
    }))
    monkeypatch.setenv("GPT_API_KEY", "synthetic-gpt-key")
    monkeypatch.setenv("GLM_API_KEY", "synthetic-glm-key")
    monkeypatch.setitem(runner.ADAPTERS, "gpt", MockProvider)
    monkeypatch.setitem(runner.ADAPTERS, "glm", MockProvider)
    monkeypatch.setattr(runner, "image_to_data_url", lambda path: "synthetic-image")
    models = ["gpt", "glm"] if enable_glm else ["gpt"]
    for name in models:
        summary = runner.run_experiment(
            experiment_path, name, "raw", output_dir=output,
        )
        assert summary["n_abstained"] == 4
        saved = (output / f"{name}-raw.config.json").read_text()
        assert "synthetic-key" not in saved
