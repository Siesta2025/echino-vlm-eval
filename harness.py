"""Fixed inference workflows that use more than one model call."""

import json


OBSERVATION_PROMPT = """For research evaluation only, inspect the provided liver
ultrasound image. Describe visible image evidence without diagnosing the disease,
using unavailable clinical information, or reading diagnostic text overlays.

Return only a JSON object with exactly these fields:
- "image_quality": "adequate", "limited", or "unreadable".
- "lesion_present": true, false, or null when unreadable.
- "findings": an array of 1 to 3 short phrases describing visible morphology.
Do not include Markdown fences or additional fields.
"""


def parse_observation(raw_text: str) -> dict:
    data = json.loads(raw_text)
    if not isinstance(data, dict) or set(data) != {
        "image_quality", "lesion_present", "findings",
    }:
        raise ValueError("Observation does not match the required schema")
    if data["image_quality"] not in ("adequate", "limited", "unreadable"):
        raise ValueError("Invalid image quality")
    if data["lesion_present"] not in (True, False, None):
        raise ValueError("Invalid lesion_present value")
    if not isinstance(data["findings"], list) or not 1 <= len(data["findings"]) <= 3:
        raise ValueError("Findings must contain one to three items")
    if not all(isinstance(item, str) for item in data["findings"]):
        raise ValueError("Each finding must be text")
    return data


def add_usage(*items: dict | None) -> dict | None:
    totals = {}
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals or None


def trace_step(name: str, result: dict) -> dict:
    return {
        "step": name,
        "raw_text": result.get("raw_text"),
        "api_status": result.get("api_status"),
        "finish_reason": result.get("finish_reason"),
        "error": result.get("error"),
        "usage": result.get("usage"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "response_id": result.get("response_id"),
    }


def run_two_pass(model, final_prompt: str, image_urls: list[str], api_call, parse_result) -> dict:
    """Observe once, then make a diagnosis using the frozen skill and observation."""
    observation_result = api_call(model, OBSERVATION_PROMPT, image_urls)
    trace = [trace_step("observe", observation_result)]
    if observation_result.get("refusal") or observation_result.get("api_status") != "completed":
        observation_result.update(parse_result(observation_result))
        observation_result["harness_trace"] = trace
        return observation_result

    try:
        observation = parse_observation(observation_result["raw_text"])
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "harness_error",
            "prediction": None,
            "raw_text": None,
            "error": f"{type(exc).__name__}: {exc}",
            "api_status": "completed",
            "usage": observation_result.get("usage"),
            "elapsed_seconds": observation_result.get("elapsed_seconds"),
            "requested_model": observation_result.get("requested_model"),
            "host": observation_result.get("host"),
            "settings": observation_result.get("settings"),
            "harness_trace": trace,
        }

    decision_prompt = (
        "The first pass produced the following untrusted visual observation. "
        "Verify it against the image, then follow the domain skill and task.\n\n"
        f"<observation>\n{json.dumps(observation, ensure_ascii=False)}\n"
        f"</observation>\n\n{final_prompt}"
    )
    final_result = api_call(model, decision_prompt, image_urls)
    final_result.update(parse_result(final_result))
    trace.append(trace_step("diagnose", final_result))
    final_result["harness_trace"] = trace
    final_result["usage"] = add_usage(
        observation_result.get("usage"), final_result.get("usage"),
    )
    final_result["elapsed_seconds"] = sum(
        value for value in (
            observation_result.get("elapsed_seconds"),
            final_result.get("elapsed_seconds"),
        ) if value is not None
    )
    return final_result
