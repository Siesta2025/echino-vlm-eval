import json
from pathlib import Path
from time import perf_counter
from openai import OpenAI
from PIL import Image
import base64


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

class Provider:
    def __init__(self, name: str, api_key: str, model_name: str, base_url: str):
        self.name = name
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def __str__(self):
        return f"{self.name} ({self.model_name})"

    def infer(self, text_prompt: str, image_data_url: str | None = None) -> dict:
        raise NotImplementedError("Each adapter implements its own API call")

    def save(self, infer_result: dict, save_path: str | Path):
        path = Path(save_path)
        line = json.dumps(infer_result, ensure_ascii=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

class OpenAIProvider(Provider):
    """OpenAI-compatible Chat Completions; protocol does not imply official hosting."""

    def __init__(self, name: str, api_key: str, model_name: str, base_url: str,
                 settings: dict | None = None):
        super().__init__(name, api_key, model_name, base_url)
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=30, max_retries=0,
        )
        # Start with the request shape verified by the local API smoke test.
        self.settings = {"stream": False, **(settings or {})}

    def infer(self, text_prompt: str, image_data_url: str | None = None) -> dict:
        content = [{"type": "text", "text": text_prompt}]
        if image_data_url is not None:
            content.append({
                "type": "image_url", "image_url": {"url": image_data_url},
            })

        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            **self.settings,
        )
        elapsed = perf_counter() - start
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        return {
            "raw_text": choice.message.content or "",
            "refusal": choice.message.refusal,
            "usage": response.usage.model_dump() if response.usage is not None else None,
            "elapsed_seconds": elapsed,
            "api_status": (
                "completed" if finish_reason == "stop"
                else "incomplete" if finish_reason == "length" else finish_reason
            ),
            "finish_reason": finish_reason,
            "incomplete_details": {"reason": "token_limit"} if finish_reason == "length" else None,
            "api_error": None,
            "response_id": response.id,
            "model": response.model,
            "requested_model": self.model_name,
            "host": self.base_url,
            "settings": {**self.settings, "image_detail": None},
        }

class GLMProvider(Provider):
    def __init__(self, name: str, api_key: str, model_name: str, base_url: str,
                 settings: dict | None = None):
        super().__init__(name, api_key, model_name, base_url)
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=30, max_retries=0,
        )
        self.settings = {"reasoning_effort": "low", "max_tokens": 2048, **(settings or {})}

    def infer(self, text_prompt: str, image_data_url: str | None = None) -> dict:
        content = [{"type": "text", "text": text_prompt}]
        if image_data_url is not None:
            content.append({
                "type": "image_url", "image_url": {"url": image_data_url},
            })

        start = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            extra_body={"provider": {"require_parameters": True}},
            **self.settings,
        )
        elapsed = perf_counter() - start
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        return {
            "raw_text": choice.message.content or "",
            "usage": response.usage.model_dump() if response.usage is not None else None,
            "elapsed_seconds": elapsed,
            "api_status": (
                "completed" if finish_reason == "stop"
                else "incomplete" if finish_reason == "length" else finish_reason
            ),
            "finish_reason": finish_reason,
            "refusal": choice.message.refusal,
            "incomplete_details": {"reason": "max_tokens"} if finish_reason == "length" else None,
            "api_error": None,
            "response_id": response.id,
            "model": response.model,
            "requested_model": self.model_name,
            "host": self.base_url,
            "upstream_provider": getattr(response, "provider", None),
            "settings": {
                **self.settings, "image_detail": None,
                "provider": {"require_parameters": True},
            },
        }
