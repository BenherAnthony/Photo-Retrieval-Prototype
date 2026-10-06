import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

PHOTO_DIR = Path("Sample Photos")
OUTPUT_FILE = Path("generated_metadata.json")

MODEL_NAME = "gemini-3.1-flash-lite"

#change range for photos to be processed here
PHOTO_IDS_TO_PROCESS = [
    f"P{i:03d}"
    for i in range(15, 16)
]

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "photo_id": {
            "type": "string"
        },
        "description": {
            "type": "string"
        },
        "location": {
            "type": "string"
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "people": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "scene": {
            "type": "string"
        },
        "location_clues": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "text_visible": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "activities": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "search_terms": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "photo_id",
        "description",
        "location",
        "objects",
        "people",
        "scene",
        "location_clues",
        "text_visible",
        "activities",
        "search_terms",
        "date"
    ]
}


def get_image_path(photo_id: str) -> Path:
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    matching_files = [
        path
        for path in PHOTO_DIR.iterdir()
        if (
            path.is_file()
            and path.stem.lower() == photo_id.lower()
            and path.suffix.lower() in supported_extensions
        )
    ]

    if not matching_files:
        raise FileNotFoundError(
            f"No image found for {photo_id} in {PHOTO_DIR.resolve()}"
        )

    if len(matching_files) > 1:
        raise RuntimeError(
            f"Multiple images found for {photo_id}: {matching_files}"
        )

    return matching_files[0]


def get_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    if suffix == ".png":
        return "image/png"

    if suffix == ".webp":
        return "image/webp"

    raise ValueError(f"Unsupported image format: {suffix}")


def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(photo_id: str, image_path: Path) -> dict:
    mime_type = get_mime_type(image_path)
    image_data = image_to_base64(image_path)

    prompt = """
Analyze this photo for a photo-retrieval prototype.

Return factual, visually grounded metadata only.

Describe what a user might remember when trying to find this image later.
Include useful natural-language search terms, but do not add facts that are not visible.

Do not invent:
- exact locations
- dates
- names or identities
- events
- relationships between people
- activities that are not visibly happening
Set location to an empty string.
-Do not infer or guess the location.
-The location will be manually added later.
Set date to an empty string.
-Do not infer or guess the date.
-The date will be manually added later.

For location_clues, include only visible clues such as:
- landmarks
- signs
- storefronts
- distinctive architecture
- recognizable landscapes
- readable text

For activities, include only activities visibly happening in the image.
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image",
                "data": image_data,
                "mime_type": mime_type
            }
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": METADATA_SCHEMA
        }
    )

    output_text = interaction.output_text

    if not output_text:
        raise RuntimeError(
            f"The model returned no text output for {photo_id}."
        )

    try:
        metadata = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON returned for {photo_id}.\n"
            f"Raw response:\n{output_text}"
        ) from error

    metadata["photo_id"] = photo_id
    metadata["location"] = ""
    metadata.setdefault("date", "")

    return metadata


def load_existing_metadata() -> list:
    if not OUTPUT_FILE.exists():
        return []

    with OUTPUT_FILE.open("r", encoding="utf-8") as input_file:
        existing_data = json.load(input_file)

    if not isinstance(existing_data, list):
        raise ValueError(
            f"{OUTPUT_FILE} must contain a JSON list."
        )

    return existing_data


def save_metadata(metadata_records: list) -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(
            metadata_records,
            output_file,
            indent=2,
            ensure_ascii=False
        )


def main():
    if not PHOTO_DIR.exists():
        raise FileNotFoundError(
            f"Photo directory not found: {PHOTO_DIR.resolve()}"
        )

    existing_metadata = load_existing_metadata()

    records_by_id = {
        record.get("photo_id"): record
        for record in existing_metadata
        if isinstance(record, dict) and record.get("photo_id")
    }

    for photo_id in PHOTO_IDS_TO_PROCESS:
        if photo_id in records_by_id:
            print(f"Skipping {photo_id}; metadata already exists.")
            continue

        image_path = get_image_path(photo_id)

        print(f"Analyzing {image_path} using {MODEL_NAME}...")

        metadata = analyze_image(photo_id, image_path)
        records_by_id[photo_id] = metadata

        save_metadata(list(records_by_id.values()))

        print(f"Saved metadata for {photo_id}.")

        time.sleep(2)

    final_records = list(records_by_id.values())
    final_records.sort(key=lambda record: record["photo_id"])

    save_metadata(final_records)

    print(
        f"Completed. Saved {len(final_records)} records to "
        f"{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()