import json


def parse_prediction(raw_text: str) -> dict:
    def failure(reason: str) -> dict:
        return {"status": "parse_error", "prediction": None, "error": reason}

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return failure("Model output is not valid JSON")
    if not isinstance(data, dict):
        return failure("Model output must be a JSON object")
    prediction = data.get("prediction")
    if prediction not in ("positive", "negative", "indeterminate"):
        return failure("Missing or invalid prediction")
    return {"status": "ok", "prediction": prediction, "error": None}
