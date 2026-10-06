import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


# =================================================
# PAGE CONFIGURATION
# =================================================

st.set_page_config(
    page_title="Vague Memory Photo Retrieval",
    page_icon="🔎",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f1117;
        }

        [data-testid="stSidebar"] {
            background-color: #1a1d27;
        }

        .main-title {
            color: #f8fafc;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #aab2c0;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }

        .collection-title {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 1rem;
        }

        .photo-details {
            background-color: #1a1f2b;
            border: 1px solid #30394a;
            border-radius: 14px;
            padding: 1.2rem 1.35rem;
            min-height: 255px;
        }

        .photo-details h3 {
            color: #ffffff;
            margin-top: 0;
            margin-bottom: 1rem;
        }

        .photo-details p {
            color: #d8dee9;
            margin: 0.5rem 0;
            font-size: 0.95rem;
        }

        .photo-details strong {
            color: #ffffff;
        }

        div[data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid #30394a;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }

        hr {
            border-color: #30394a;
        }

        .match-text {
            color: #aab2c0;
            font-size: 0.88rem;
            margin-top: 0.7rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =================================================
# PATHS AND GEMINI CONFIGURATION
# =================================================

BASE_DIR = Path(__file__).parent
SAMPLE_PHOTOS_DIR = BASE_DIR / "Sample Photos"
METADATA_JSON_PATH = BASE_DIR / "generated_metadata.json"

MODEL_NAME = "gemini-3.1-flash-lite"

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    st.error(
        "Gemini API key is missing. Configure GEMINI_API_KEY in .env "
        "or Streamlit Secrets."
    )
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# =================================================
# QUERY SCHEMA
# =================================================

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "location_mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mentioned_place": {"type": "string"},
                    "normalized_location": {"type": "string"},
                    "place_level": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["include", "exclude"]
                    },
                    "confidence": {"type": "number"}
                },
                "required": [
                    "mentioned_place",
                    "normalized_location",
                    "place_level",
                    "relation",
                    "confidence"
                ]
            }
        },
        "date_clues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["exact", "range", "approximate"]
                    },
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["include", "exclude"]
                    },
                    "original_text": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": [
                    "type",
                    "start_date",
                    "end_date",
                    "relation",
                    "original_text",
                    "confidence"
                ]
            }
        },
        "people_clues": {
            "type": "object",
            "properties": {
                "include_people": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "exclude_people": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "include_relationships": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "exclude_relationships": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "include_count": {
                    "type": "integer",
                    "nullable": True
                },
                "exclude_count": {
                    "type": "integer",
                    "nullable": True
                },
                "appearance_or_group_clues": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "has_people_constraint": {"type": "boolean"}
            },
            "required": [
                "include_people",
                "exclude_people",
                "include_relationships",
                "exclude_relationships",
                "include_count",
                "exclude_count",
                "appearance_or_group_clues",
                "has_people_constraint"
            ]
        },
        "object_clues": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["include", "exclude"]
        },
        "activity_clues": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["include", "exclude"]
        },
        "free_text_clues": {
            "type": "object",
            "properties": {
                "include": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["include", "exclude"]
        },
        "needs_clarification": {"type": "boolean"}
    },
    "required": [
        "location_mentions",
        "date_clues",
        "people_clues",
        "object_clues",
        "activity_clues",
        "free_text_clues",
        "needs_clarification"
    ]
}


# =================================================
# SESSION STATE
# =================================================

DEFAULT_STATE = {
    "initial_candidates": None,
    "selected_collection": None,
    "candidate_groups": None,
    "follow_up": None,
    "refined_results": None,
    "original_memory": "",
    "ai_logs": [],
    "latest_query_json": None,
    "latest_follow_up_json": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =================================================
# DATA LOADING
# =================================================

def list_to_text(value):
    if isinstance(value, list):
        return "|".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "")


def load_photos():
    if not METADATA_JSON_PATH.exists():
        st.error(
            f"generated_metadata.json was not found at: "
            f"{METADATA_JSON_PATH}"
        )
        st.stop()

    with METADATA_JSON_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        st.error("generated_metadata.json must contain a JSON list.")
        st.stop()

    rows = []

    for record in records:
        photo_id = str(record.get("photo_id", "")).strip()

        if not photo_id:
            continue

        rows.append({
            "photo_id": photo_id,
            "file_name": f"{photo_id}.jpg",
            "date": str(record.get("date", "") or ""),
            "location": str(record.get("location", "") or ""),
            "description": str(record.get("description", "") or ""),
            "objects": list_to_text(record.get("objects", [])),
            "people": list_to_text(record.get("people", [])),
            "scene": str(record.get("scene", "") or ""),
            "location_clues": list_to_text(
                record.get("location_clues", [])
            ),
            "text_visible": list_to_text(
                record.get("text_visible", [])
            ),
            "activities": list_to_text(
                record.get("activities", [])
            ),
            "search_terms": list_to_text(
                record.get("search_terms", [])
            )
        })

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        st.error("No photo records were loaded.")
        st.stop()

    return dataframe


photos = load_photos()


# =================================================
# FORMATTING AND VALUE HELPERS
# =================================================

def split_values(value):
    if value is None:
        return []

    return [
        item.strip().lower()
        for item in str(value).split("|")
        if item.strip()
    ]


def format_values(value):
    return ", ".join(
        item.strip()
        for item in str(value or "").split("|")
        if item.strip()
    )


def get_available_people():
    values = set()

    for value in photos["people"]:
        values.update(split_values(value))

    return sorted(values)


def get_available_locations():
    values = set()

    for value in photos["location"]:
        location = str(value).strip()

        if location:
            values.add(location)

    return sorted(values)


def clean_json_response(response_text):
    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]

    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


# =================================================
# GEMINI CALL
# =================================================

def call_gemini_json(prompt, call_type):
    max_attempts = 4
    base_delay_seconds = 2

    for attempt in range(max_attempts):
        try:
            request_started_at = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": QUERY_SCHEMA
                }
            )

            parsed_response = json.loads(
                clean_json_response(response.text)
            )

            usage = getattr(response, "usage_metadata", None)

            prompt_tokens = getattr(
                usage,
                "prompt_token_count",
                0
            ) or 0

            output_tokens = getattr(
                usage,
                "candidates_token_count",
                0
            ) or 0

            thinking_tokens = getattr(
                usage,
                "thoughts_token_count",
                0
            ) or 0

            total_tokens = getattr(
                usage,
                "total_token_count",
                0
            ) or 0

            st.session_state.ai_logs.append({
                "timestamp": request_started_at,
                "call_type": call_type,
                "model": MODEL_NAME,
                "status": "Success",
                "attempt": attempt + 1,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_tokens,
                "error": ""
            })

            return parsed_response

        except Exception as error:
            error_message = str(error)

            temporary_error = (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            final_attempt = attempt == max_attempts - 1

            if not temporary_error or final_attempt:
                st.session_state.ai_logs.append({
                    "timestamp": time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "call_type": call_type,
                    "model": MODEL_NAME,
                    "status": "Failed",
                    "attempt": attempt + 1,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "total_tokens": 0,
                    "error": error_message
                })

                raise error

            wait_seconds = (
                base_delay_seconds * (2 ** attempt)
                + random.uniform(0, 1)
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Gemini could not complete the request.")


# =================================================
# QUERY ANALYSIS
# =================================================

def extract_query_clues(user_query):
    prompt = f"""
You analyze natural-language photo retrieval queries.

User query:
"{user_query}"

Available canonical people:
{get_available_people()}

Available canonical photo locations:
{get_available_locations()}

Return only JSON matching the provided schema.

Rules:

1. Extract only clues stated or reasonably implied.
2. Do not invent people, dates, objects, or events.
3. Normalize neighborhood or locality names to their broader city when
   reasonably known. For example, Marthahalli or Marathahalli may normalize
   to Bengaluru. Preserve the original place in mentioned_place.
4. Use relation "exclude" for negated language such as:
   "without Rahul", "not Goa", "except screenshots".
5. For date ranges, populate start_date and end_date using YYYY-MM-DD.
6. If only a month is known, use the first and last day of that month.
7. If a date cannot be resolved reliably, use empty dates and type
   "approximate".
8. Keep explicitly named people in include_people or exclude_people.
9. "Group photos" is an include_relationships clue.
10. Do not create a location clue merely because an image object has a name.
11. Use free_text_clues only for clues that do not fit a more specific field.
12. Set needs_clarification true only when a missing detail is likely to
    materially change the candidate results.
"""

    parsed = call_gemini_json(
        prompt,
        "Query clue extraction"
    )

    st.session_state.latest_query_json = parsed

    return parsed


# =================================================
# MATCHING UTILITIES
# =================================================

def contains_value(row, columns, term):
    term = str(term).strip().lower()

    if not term:
        return False

    combined = " ".join(
        str(row.get(column, "")).lower()
        for column in columns
    )

    return term in combined


def row_people(row):
    return split_values(row.get("people", ""))


def date_range_overlap(photo_date, start_date, end_date):
    photo_date = str(photo_date or "").strip()
    start_date = str(start_date or "").strip()
    end_date = str(end_date or "").strip()

    if not photo_date or not start_date or not end_date:
        return False

    if len(photo_date) == 4:
        photo_start = f"{photo_date}-01-01"
        photo_end = f"{photo_date}-12-31"

    elif len(photo_date) == 7:
        year, month = photo_date.split("-")
        next_month = int(month) % 12 + 1
        next_year = int(year) + (1 if month == "12" else 0)
        last_day = (
            pd.Timestamp(
                year=next_year,
                month=next_month,
                day=1
            ) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        photo_start = f"{photo_date}-01"
        photo_end = last_day

    else:
        photo_start = photo_date
        photo_end = photo_date

    return photo_start <= end_date and photo_end >= start_date


def add_reason(results, mask, points, reason):
    results.loc[mask, "match_score"] += points
    results.loc[mask, "match_reasons"] += f"{reason};"


def score_candidates(photo_data, clues):
    results = photo_data.copy()
    results["match_score"] = 0.0
    results["match_reasons"] = ""

    max_score = 0.0

    location_mentions = clues.get("location_mentions", [])
    date_clues = clues.get("date_clues", [])
    people_clues = clues.get("people_clues", {})
    object_clues = clues.get("object_clues", {})
    activity_clues = clues.get("activity_clues", {})
    free_text_clues = clues.get("free_text_clues", {})

    include_locations = [
        item.get("normalized_location", "").strip().lower()
        for item in location_mentions
        if item.get("relation") == "include"
    ]

    exclude_locations = [
        item.get("normalized_location", "").strip().lower()
        for item in location_mentions
        if item.get("relation") == "exclude"
    ]

    for location in include_locations:
        if location:
            max_score += 25
            mask = results["location"].str.lower() == location
            add_reason(results, mask, 25, f"Location: {location}")

    for location in exclude_locations:
        if location:
            mask = results["location"].str.lower() == location
            results.loc[mask, "match_score"] = -1000
            results.loc[mask, "match_reasons"] += (
                f"Excluded location: {location};"
            )

    include_people = [
        str(item).strip().lower()
        for item in people_clues.get("include_people", [])
        if str(item).strip()
    ]

    exclude_people = [
        str(item).strip().lower()
        for item in people_clues.get("exclude_people", [])
        if str(item).strip()
    ]

    for person in include_people:
        max_score += 30
        mask = results["people"].apply(
            lambda value: person in split_values(value)
        )
        add_reason(results, mask, 30, f"Person: {person}")

    for person in exclude_people:
        mask = results["people"].apply(
            lambda value: person in split_values(value)
        )
        results.loc[mask, "match_score"] = -1000
        results.loc[mask, "match_reasons"] += (
            f"Excluded person: {person};"
        )

    include_relationships = [
        str(item).strip().lower()
        for item in people_clues.get("include_relationships", [])
    ]

    for relationship in include_relationships:
        max_score += 10
        mask = results.apply(
            lambda row: relationship in (
                " ".join(
                    split_values(row.get("people", ""))
                    + split_values(row.get("description", ""))
                    + split_values(row.get("scene", ""))
                )
            ),
            axis=1
        )
        add_reason(results, mask, 10, f"Group clue: {relationship}")

    include_objects = object_clues.get("include", [])
    exclude_objects = object_clues.get("exclude", [])

    searchable_columns = [
        "description",
        "objects",
        "scene",
        "location_clues",
        "text_visible",
        "activities",
        "search_terms"
    ]

    for object_term in include_objects:
        max_score += 15
        mask = results.apply(
            lambda row: contains_value(
                row,
                searchable_columns,
                object_term
            ),
            axis=1
        )
        add_reason(results, mask, 15, f"Object: {object_term}")

    for object_term in exclude_objects:
        mask = results.apply(
            lambda row: contains_value(
                row,
                searchable_columns,
                object_term
            ),
            axis=1
        )
        results.loc[mask, "match_score"] = -1000
        results.loc[mask, "match_reasons"] += (
            f"Excluded object: {object_term};"
        )

    for activity in activity_clues.get("include", []):
        max_score += 15
        mask = results.apply(
            lambda row: contains_value(
                row,
                ["activities", "description", "scene"],
                activity
            ),
            axis=1
        )
        add_reason(results, mask, 15, f"Activity: {activity}")

    for activity in activity_clues.get("exclude", []):
        mask = results.apply(
            lambda row: contains_value(
                row,
                ["activities", "description", "scene"],
                activity
            ),
            axis=1
        )
        results.loc[mask, "match_score"] = -1000
        results.loc[mask, "match_reasons"] += (
            f"Excluded activity: {activity};"
        )

    for text_term in free_text_clues.get("include", []):
        max_score += 12
        mask = results.apply(
            lambda row: contains_value(
                row,
                searchable_columns,
                text_term
            ),
            axis=1
        )
        add_reason(results, mask, 12, f"Clue: {text_term}")

    for text_term in free_text_clues.get("exclude", []):
        mask = results.apply(
            lambda row: contains_value(
                row,
                searchable_columns,
                text_term
            ),
            axis=1
        )
        results.loc[mask, "match_score"] = -1000
        results.loc[mask, "match_reasons"] += (
            f"Excluded clue: {text_term};"
        )

    for date_clue in date_clues:
        start_date = date_clue.get("start_date", "")
        end_date = date_clue.get("end_date", "")
        relation = date_clue.get("relation", "include")

        if not start_date or not end_date:
            continue

        mask = results["date"].apply(
            lambda value: date_range_overlap(
                value,
                start_date,
                end_date
            )
        )

        if relation == "include":
            max_score += 20
            add_reason(
                results,
                mask,
                20,
                f"Date: {start_date} to {end_date}"
            )
        else:
            results.loc[mask, "match_score"] = -1000
            results.loc[mask, "match_reasons"] += (
                f"Excluded date: {start_date} to {end_date};"
            )

    if max_score <= 0:
        max_score = 1

    results["confidence"] = (
        results["match_score"].clip(lower=0) / max_score * 100
    ).clip(upper=100).round(0)

    results = results[results["match_score"] > 0]
    results = results.sort_values(
        by=["match_score", "photo_id"],
        ascending=[False, True]
    )

    return results


# =================================================
# AMBIGUITY AND COLLECTIONS
# =================================================

def should_group_by_location(candidates, clues):
    location_mentions = clues.get("location_mentions", [])
    people_clues = clues.get("people_clues", {})

    has_explicit_location = any(
        item.get("relation") == "include"
        for item in location_mentions
    )

    has_people_constraint = people_clues.get(
        "has_people_constraint",
        False
    )

    if has_explicit_location or has_people_constraint:
        return False

    return candidates["location"].nunique() > 1


def get_location_collections(candidates):
    groups = {}

    for location, group in candidates.groupby(
        "location",
        dropna=False
    ):
        location_name = str(location).strip() or "Unknown location"
        groups[location_name] = group.copy()

    return groups


def get_people_collections(candidates):
    groups = {
        "Rahul and Priya together": candidates.copy()
    }

    return groups


def get_follow_up_question(candidates, user_query, clues):
    summary = []

    for _, row in candidates.head(20).iterrows():
        summary.append({
            "photo_id": row["photo_id"],
            "location": row["location"],
            "date": row["date"],
            "people": row["people"],
            "description": row["description"],
            "objects": row["objects"],
            "scene": row["scene"],
            "activities": row["activities"]
        })

    prompt = f"""
You choose at most one high-value follow-up question for photo retrieval.

Original query:
"{user_query}"

Already extracted clues:
{json.dumps(clues, indent=2)}

Candidate photos:
{json.dumps(summary, indent=2)}

Return only valid JSON in this structure:

{{
  "ask_question": false,
  "question": "",
  "field": "none",
  "options": []
}}

Allowed fields:
- location
- people_group
- date
- object
- setting
- none

Rules:
1. Ask only one question.
2. Ask only if it will materially reduce the candidates.
3. Do not ask for a clue the user already provided.
4. For ambiguous locations, ask which location.
5. For Rahul/Priya-type queries, ask whether the user means only those people
   or a larger group including them.
6. Keep the question short and neutral.
7. If results are already sufficiently strong, do not ask.
"""

    follow_up_schema = {
        "type": "object",
        "properties": {
            "ask_question": {"type": "boolean"},
            "question": {"type": "string"},
            "field": {
                "type": "string",
                "enum": [
                    "location",
                    "people_group",
                    "date",
                    "object",
                    "setting",
                    "none"
                ]
            },
            "options": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": [
            "ask_question",
            "question",
            "field",
            "options"
        ]
    }

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": follow_up_schema
        }
    )

    parsed = json.loads(clean_json_response(response.text))
    st.session_state.latest_follow_up_json = parsed

    return parsed


# =================================================
# DISPLAY
# =================================================

def show_photo_cards(photo_data):
    if photo_data.empty:
        st.warning("No photo matches the selected clues.")
        return

    for _, photo in photo_data.iterrows():
        image_path = SAMPLE_PHOTOS_DIR / str(photo["file_name"])

        image_column, details_column = st.columns(
            [1.05, 1.45],
            gap="large"
        )

        with image_column:
            if image_path.exists():
                st.image(
                    str(image_path),
                    caption=str(photo["photo_id"]),
                    use_container_width=True
                )
            else:
                st.error(f"Missing image: {photo['file_name']}")

        with details_column:
            st.markdown(
                f"""
                <div class="photo-details">
                    <h3>{photo["photo_id"]}</h3>
                    <p><strong>Date:</strong> {photo["date"]}</p>
                    <p><strong>Location:</strong> {photo["location"]}</p>
                    <p><strong>People:</strong> {
                        format_values(photo["people"])
                    }</p>
                    <p><strong>Scene:</strong> {
                        format_values(photo["scene"])
                    }</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Match confidence",
                f"{int(photo['confidence'])}%"
            )

            reasons = [
                item.strip()
                for item in str(photo["match_reasons"]).split(";")
                if item.strip()
            ]

            if reasons:
                st.markdown(
                    "Matched because: " + ", ".join(reasons),
                    unsafe_allow_html=True
                )

            visible_text = format_values(photo["text_visible"])

            if visible_text:
                st.info(f"Visible text: {visible_text}")

        st.divider()


def reset_search_state():
    st.session_state.initial_candidates = None
    st.session_state.selected_collection = None
    st.session_state.candidate_groups = None
    st.session_state.follow_up = None
    st.session_state.refined_results = None
    st.session_state.original_memory = ""
    st.session_state.latest_query_json = None
    st.session_state.latest_follow_up_json = None


# =================================================
# HEADER AND NAVIGATION
# =================================================

st.markdown(
    '<div class="main-title">🔎 Vague Memory Photo Retrieval</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Find old photos using partial memories, visual clues,
        people, locations, dates, and text you remember seeing.
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Search photos",
        "Browse sample gallery",
        "View metadata",
        "AI JSON & Logs"
    ]
)


# =================================================
# SEARCH PAGE
# =================================================

if page == "Search photos":
    st.subheader("Describe the photo you remember")

    user_memory = st.text_area(
        "What do you remember?",
        placeholder=(
            "Example: Show me the blue building. "
            "I am not sure whether it was in Bengaluru or Goa."
        ),
        height=120
    )

    if st.button("Search my photos", type="primary"):
        reset_search_state()

        if not user_memory.strip():
            st.warning("Please describe what you remember.")
        else:
            with st.spinner("Understanding your memory..."):
                try:
                    clues = extract_query_clues(user_memory)
                    candidates = score_candidates(photos, clues)

                    st.session_state.initial_candidates = candidates
                    st.session_state.original_memory = user_memory

                except Exception as error:
                    error_message = str(error)

                    if (
                        "503" in error_message
                        or "UNAVAILABLE" in error_message
                    ):
                        st.warning(
                            "The AI service is temporarily busy. "
                            "Please wait and try again."
                        )
                    elif (
                        "429" in error_message
                        or "RESOURCE_EXHAUSTED" in error_message
                    ):
                        st.warning(
                            "The Gemini API quota has been reached."
                        )
                    else:
                        st.error(f"Search failed: {error_message}")

    candidates = st.session_state.initial_candidates

    if candidates is not None:
        if candidates.empty:
            st.warning(
                "No matching photos were found. Try a broader memory."
            )
        else:
            clues = st.session_state.latest_query_json

            if should_group_by_location(candidates, clues):
                st.markdown(
                    '<div class="collection-title">'
                    'Possible location collections'
                    '</div>',
                    unsafe_allow_html=True
                )

                location_groups = get_location_collections(candidates)

                for location, group in location_groups.items():
                    st.markdown(
                        f"### {location} "
                        f"({len(group)} matching photo(s))"
                    )
                    show_photo_cards(group.head(5))

                location_options = list(location_groups.keys())

                selected_location = st.radio(
                    "Which location was it?",
                    location_options
                )

                selected_collection = location_groups[selected_location]

                if st.button(
                    "Use this location",
                    type="primary"
                ):
                    st.session_state.selected_collection = (
                        selected_collection
                    )
                    st.session_state.refined_results = None

            else:
                st.markdown(
                    '<div class="collection-title">'
                    'Ranked photo results'
                    '</div>',
                    unsafe_allow_html=True
                )

                show_photo_cards(candidates.head(10))

                if st.button(
                    "Use these results",
                    type="primary"
                ):
                    st.session_state.selected_collection = candidates
                    st.session_state.refined_results = None

    selected_collection = st.session_state.selected_collection

    if selected_collection is not None:
        st.divider()

        st.subheader("Optional refinement")

        if st.button("Find one useful follow-up question"):
            with st.spinner("Finding the most useful question..."):
                try:
                    st.session_state.follow_up = (
                        get_follow_up_question(
                            selected_collection,
                            st.session_state.original_memory,
                            st.session_state.latest_query_json
                        )
                    )
                except Exception as error:
                    st.error(
                        f"Could not generate follow-up question: {error}"
                    )

        follow_up = st.session_state.follow_up

        if follow_up and follow_up.get("ask_question"):
            st.write(follow_up["question"])

            options = follow_up.get("options", [])

            if not options:
                options = ["Not sure"]

            answer = st.selectbox(
                "Your answer",
                ["Not sure"] + options
            )

            if st.button("Apply follow-up"):
                refined = selected_collection.copy()

                field = follow_up.get("field")

                if field == "location":
                    refined = refined[
                        refined["location"].str.contains(
                            answer,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ]

                elif field == "object":
                    refined = refined[
                        refined.apply(
                            lambda row: contains_value(
                                row,
                                [
                                    "description",
                                    "objects",
                                    "scene",
                                    "search_terms"
                                ],
                                answer
                            ),
                            axis=1
                        )
                    ]

                elif field == "date":
                    refined = refined[
                        refined["date"].str.contains(
                            answer,
                            case=False,
                            na=False,
                            regex=False
                        )
                    ]

                elif field == "setting":
                    refined = refined[
                        refined.apply(
                            lambda row: contains_value(
                                row,
                                ["scene", "description"],
                                answer
                            ),
                            axis=1
                        )
                    ]

                st.session_state.refined_results = refined

        current_results = (
            st.session_state.refined_results
            if st.session_state.refined_results is not None
            else selected_collection
        )

        st.subheader("Results")

        show_photo_cards(current_results)

        if not current_results.empty:
            selected_photo = st.selectbox(
                "Select the photo you were looking for",
                current_results["photo_id"].tolist()
            )

            if st.button("This is the photo"):
                st.success(
                    f"Retrieval completed. You selected {selected_photo}."
                )
                st.balloons()


# =================================================
# GALLERY PAGE
# =================================================

elif page == "Browse sample gallery":
    st.subheader("Browse the sample photo library")

    keyword = st.text_input(
        "Search photos",
        placeholder="Try blue, building, Goa, flowers, birthday..."
    ).strip().lower()

    filtered = photos.copy()

    if keyword:
        searchable_columns = [
            "photo_id",
            "date",
            "location",
            "description",
            "objects",
            "people",
            "scene",
            "location_clues",
            "text_visible",
            "activities",
            "search_terms"
        ]

        combined_text = filtered[searchable_columns].astype(str).agg(
            " ".join,
            axis=1
        ).str.lower()

        filtered = filtered[
            combined_text.str.contains(
                keyword,
                na=False,
                regex=False
            )
        ]

    st.caption(
        f"Showing {len(filtered)} of {len(photos)} sample photos"
    )

    show_photo_cards(filtered)


# =================================================
# METADATA PAGE
# =================================================

elif page == "View metadata":
    reload_column, info_column = st.columns()[1][4]

    with reload_column:
        if st.button("Reload metadata"):
            st.rerun()

    with info_column:
        metadata_last_updated = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(METADATA_JSON_PATH.stat().st_mtime)
        )

        st.caption(
            f"Metadata JSON last modified: {metadata_last_updated}"
        )

    st.subheader("Generated photo metadata")

    st.dataframe(
        photos,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.write("Metadata path:", str(METADATA_JSON_PATH))
    st.write("Metadata exists:", METADATA_JSON_PATH.exists())
    st.write("Photo folder:", str(SAMPLE_PHOTOS_DIR))
    st.write("Photo folder exists:", SAMPLE_PHOTOS_DIR.exists())


# =================================================
# AI LOG PAGE
# =================================================

elif page == "AI JSON & Logs":
    st.subheader("AI JSON & Token Logs")

    query_tab, follow_up_tab, token_tab = st.tabs([
        "Query JSON",
        "Follow-up JSON",
        "Token Logs"
    ])

    with query_tab:
        if st.session_state.latest_query_json:
            st.json(st.session_state.latest_query_json)
        else:
            st.info("No query analysis has been run yet.")

    with follow_up_tab:
        if st.session_state.latest_follow_up_json:
            st.json(st.session_state.latest_follow_up_json)
        else:
            st.info("No follow-up analysis has been run yet.")

    with token_tab:
        if st.session_state.ai_logs:
            logs_dataframe = pd.DataFrame(
                st.session_state.ai_logs
            )

            metrics = st.columns(4)

            metrics.metric([0]
                "Prompt tokens",
                int(logs_dataframe["prompt_tokens"].sum())
            )

            metrics.metric([1]
                "Output tokens",
                int(logs_dataframe["output_tokens"].sum())
            )

            metrics.metric([2]
                "Thinking tokens",
                int(logs_dataframe["thinking_tokens"].sum())
            )

            metrics.metric([3]
                "Total tokens",
                int(logs_dataframe["total_tokens"].sum())
            )

            st.dataframe(
                logs_dataframe,
                use_container_width=True,
                hide_index=True
            )

            if st.button("Clear AI logs"):
                st.session_state.ai_logs = []
                st.session_state.latest_query_json = None
                st.session_state.latest_follow_up_json = None
                st.rerun()
        else:
            st.info("No Gemini API calls have been made yet.")