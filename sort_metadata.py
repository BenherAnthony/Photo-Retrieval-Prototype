import json
from pathlib import Path

INPUT_FILE = Path("generated_metadata.json")

with INPUT_FILE.open("r", encoding="utf-8") as f:
    records = json.load(f)

records.sort(key=lambda record: int(record["photo_id"][1:]))

with INPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"Sorted {len(records)} records by photo_id.")