import json
from pathlib import Path


class DataLoader:
    def __init__(self, name: str, data_path: str | Path):
        self.name = name
        self.data_path = Path(data_path)

    def load_data(self) -> list[dict]:
        with self.data_path.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        return data
