import json
from collections import Counter
from pathlib import Path


class Evaluator:
    def __init__(self, name: str, gt_data: list[dict], infer_result_path: str | Path,
                 eval_result_path: str | Path):
        self.name = name
        self.gt_data = gt_data
        self.infer_result_path = Path(infer_result_path)
        self.eval_result_path = Path(eval_result_path)

    def evaluate(self, overwrite: bool = False) -> dict:
        with self.infer_result_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        gold_by_id = {sample["case_id"]: sample["gold_label"] for sample in self.gt_data}
        records_by_id = {record["case_id"]: record for record in records}
        if len(gold_by_id) != len(self.gt_data) or len(records_by_id) != len(records):
            raise ValueError("Duplicate case_id")
        if not records_by_id.keys() <= gold_by_id.keys():
            raise ValueError("Unknown case_id")

        results = []
        confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        confusion_keys = {
            ("positive", "positive"): "tp", ("negative", "negative"): "tn",
            ("negative", "positive"): "fp", ("positive", "negative"): "fn",
        }
        for case_id, gold in gold_by_id.items():
            record = records_by_id.get(case_id, {
                "status": "missing", "prediction": None,
            })
            status, prediction = record["status"], record["prediction"]
            if status == "missing":
                comparison = "missing"
            elif status != "ok":
                comparison = "failed"
            elif prediction == "indeterminate":
                comparison = "abstained"
            else:
                comparison = "correct" if prediction == gold else "incorrect"
                confusion[confusion_keys[(gold, prediction)]] += 1
            results.append({
                "case_id": case_id, "gold_label": gold,
                "raw_text": record.get("raw_text"), "prediction": prediction,
                "status": status, "comparison": comparison,
            })

        counts = Counter(result["comparison"] for result in results)
        status_counts = Counter(record["status"] for record in records)
        n_expected = len(self.gt_data)
        n_classified = counts["correct"] + counts["incorrect"]
        summary = {
            "name": self.name,
            "n_expected": n_expected, "n_missing": counts["missing"],
            "status_counts": dict(status_counts),
            "n_abstained": counts["abstained"], "n_classified": n_classified,
            "coverage": n_classified / n_expected if n_expected else None,
            "accuracy_on_classified": counts["correct"] / n_classified if n_classified else None,
            "correct_over_all": counts["correct"] / n_expected if n_expected else None,
            "confusion_matrix": confusion,
        }

        self.eval_result_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        with self.eval_result_path.open(mode, encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return summary
