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
    "latest_follow_up_json": None,
    "gallery_selected_photo": None,
    "clarification_history": [],
    "clarification_round": 0
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

Examples:

Query: "Show me photos of a house"
Expected object_clues.include:
["house", "building"]

Query: "Find the photo with my dog"
Expected object_clues.include:
["dog", "animal"]

Query: "I remember a car near a building"
Expected object_clues.include:
["car", "vehicle", "building"]

Query: "The photo was taken at a cafe"
Expected free_text_clues.include:
["cafe"]

Query: "Show me the restaurant photo without a dog"
Expected free_text_clues.include:
["restaurant"]
Expected object_clues.exclude:
["dog", "animal"]

Query: "Show me Rahul at home"
Expected people_clues.include_people:
["rahul"]
Expected object_clues.include:
["house", "building"]

Rules:

1. Extract only clues stated or reasonably implied by the user's query.
2. Do not invent people, dates, locations, objects, activities, or events.
3. Separate clues by meaning:
   - Objects are visible things, such as car, house, building, dog, cat,
     animal, phone, medicine, flower, food, or laptop.
   - Activities are actions or events, such as eating, driving, swimming,
     studying, shopping, or birthday.
   - Locations are places or geographic areas, such as Goa, Bengaluru,
     beach, restaurant, cafe, or park, when the wording indicates a place
     rather than merely an object.
   - Free-text clues are concepts that do not fit those categories.
   - Activities are actions, such as eating, driving, walking, swimming,
     playing, studying, shopping, or celebrating.
   - Setting or scene clues describe the environment, such as indoors,
     outdoors, inside, outside, room, kitchen, bedroom, street, park, or beach.
   - Because the schema has no dedicated setting_clues field, put setting or
     scene clues in free_text_clues.include or exclude.
   - Never put indoors, outdoors, inside, or outside in activity_clues unless
     the user explicitly describes an activity occurring there.
    -Classify indoor, indoors, inside, interior, living room, bedroom,
    kitchen, office, outdoor, outdoors, outside, exterior, park, street,
    garden, and beach as scene or setting clues.
    Do not place these only in free_text_clues.
4. Normalize common synonyms and related retrieval terms into canonical
   searchable labels, but do not claim that the user explicitly said every
   synonym.
5. Use the following canonical concept expansions when the user's wording
   clearly refers to the concept:
   - house, home, residence, or bungalow:
     include "house" and "building" in object_clues.include.
   - building, apartment, office, or structure:
     include "building" in object_clues.include.
   - car, automobile, vehicle, sedan, SUV, taxi, or van:
     include "car" and "vehicle" in object_clues.include.
   - dog or puppy:
     include "dog" and "animal" in object_clues.include.
   - cat or kitten:
     include "cat" and "animal" in object_clues.include.
   - dog, cat, bird, or another clearly named animal:
     include the specific animal and "animal" in object_clues.include.
   - cafe, café, coffee shop, restaurant, or eatery:
     treat it as a location or setting clue when it refers to where the
     photo was taken, and include "cafe" or "restaurant" in the most
     appropriate field. Do not add both unless the query supports both.
6. Preserve the user's original wording in free_text_clues only when it
   contains useful information that is lost by normalization.
7. Do not expand unrelated words. For example, do not turn "wooden house"
   into "hotel", "villa", or "building with a car".
8. If a concept is negated, put the canonical terms in the relevant exclude
   field. For example, "without a car" puts "car" and "vehicle" in
   object_clues.exclude.
9. Normalize neighborhood or locality names to their broader city when
   reasonably known. Preserve the original place in mentioned_place.
10. Use relation "exclude" for negated language such as:
    "without Rahul", "not Goa", or "except screenshots".
11. For date ranges, populate start_date and end_date using YYYY-MM-DD.
12. If only a month is known, use the first and last day of that month.
13. If a date cannot be resolved reliably, use empty dates and type
    "approximate".
14. Put a term in include_people or exclude_people only if it is an
    explicitly named person and appears in the available canonical people
    list.
15. Never put generic terms such as someone, person, people, adult, man,
    woman, child, group, friends, or family in include_people or
    exclude_people. Put them in appearance_or_group_clues instead.
16. "Group photos" is an include_relationships clue.
17. Do not create a location clue merely because an image object has a name.
18. Use free_text_clues only for clues that do not fit a more specific field.
19. Set needs_clarification true only when a missing detail is likely to
    materially change the candidate results.
20. Return each canonical term at most once in each include or exclude list.
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

def filter_by_clarification(candidates, answer):
    normalized_answer = answer.strip().lower()


    if normalized_answer in [
        "indoor",
        "indoors",
        "inside",
        "interior"
    ]:
        terms = [
            "indoor",
            "indoors",
            "inside",
            "interior",
            "living room",
            "bedroom",
            "kitchen",
            "office"
        ]
    elif normalized_answer in [
        "outdoor",
        "outdoors",
        "outside",
        "exterior"
    ]:
        terms = [
            "outdoor",
            "outdoors",
            "outside",
            "exterior",
            "park",
            "street",
            "garden",
            "beach"
        ]
    else:
        return candidates


    def row_matches(row):
        row_text = " ".join(
            str(row.get(column, ""))
            for column in [
                "scene",
                "description",
                "objects",
                "search_terms",
                "activity",
                "location"
            ]
        ).lower()


        return any(term in row_text for term in terms)


    return candidates[
        candidates.apply(row_matches, axis=1)
    ]

def clue_already_present(answer, query_json):
    answer_normalized = answer.strip().lower()


    if not query_json:
        return False


    searchable_values = []


    for key in [
        "location_mentions",
        "date_clues",
        "activity_clues",
        "free_text_clues"
    ]:
        value = query_json.get(key, {})


        if isinstance(value, dict):
            for subvalue in value.values():
                if isinstance(subvalue, list):
                    searchable_values.extend(subvalue)
        elif isinstance(value, list):
            searchable_values.extend(value)


    object_clues = query_json.get("object_clues", {})


    if isinstance(object_clues, dict):
        for subvalue in object_clues.values():
            if isinstance(subvalue, list):
                searchable_values.extend(subvalue)


    searchable_text = " ".join(
        str(value).lower()
        for value in searchable_values
    )


    return answer_normalized in searchable_text

def contains_value(row, columns, term):
    term = str(term).strip().lower()


    if not term:
        return False


    combined = " ".join(
        str(row.get(column, "")).lower()
        for column in columns
    )


    synonyms = {
        "dog": [
            "dog",
            "puppy",
            "animal",
            "pet"
        ],
        "animal": [
            "dog",
            "puppy",
            "animal",
            "pet"
        ],
        "cat": [
            "cat",
            "kitten",
            "animal",
            "pet"
        ],
        "indoors": [
            "indoor",
            "indoors",
            "inside",
            "interior",
            "room",
            "kitchen",
            "bedroom",
            "living room"
        ],
        "indoor": [
            "indoor",
            "indoors",
            "inside",
            "interior",
            "room",
            "kitchen",
            "bedroom",
            "living room"
        ],
        "Bengaluru": [
            "Bangalore",
            "Bengaluru",
            "Banglore"
        ],
        "Bangalore": [
            "Bangalore",
            "Bengaluru",
            "Banglore"
        ],
        "outdoors": [
            "outdoor",
            "outdoors",
            "outside",
            "exterior",
            "street",
            "park",
            "beach"
        ],
        "outdoor": [
            "outdoor",
            "outdoors",
            "outside",
            "exterior",
            "street",
            "park",
            "beach"
        ],
        "car": [
            "car",
            "vehicle",
            "automobile",
            "sedan",
            "suv",
            "hatchback",
            "van",
            "taxi"
        ],
        "vehicle": [
            "car",
            "vehicle",
            "automobile",
            "sedan",
            "suv",
            "hatchback",
            "van",
            "taxi"
        ],
        "house": [
            "house",
            "home",
            "building",
            "residence",
            "bungalow"
        ],
        "home": [
            "house",
            "home",
            "building",
            "residence",
            "bungalow"
        ],
        "building": [
            "house",
            "home",
            "building",
            "residence",
            "bungalow"
        ]
    }


    search_terms = synonyms.get(term, [term])


    return any(
        search_term in combined
        for search_term in search_terms
    )

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
    #free_text_clues = clues.get("free_text_clues", {})
    free_text_clues = {
        "include": [],
        "exclude": []
    }
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

    generic_people_terms = {
        "someone",
        "person",
        "people",
        "adult",
        "man",
        "woman",
        "child",
        "kid",
        "boy",
        "girl",
        "gentleman",
        "lady",
        "group",
        "friends",
        "family"
    }


    include_people = [
        str(item).strip().lower()
        for item in people_clues.get("include_people", [])
        if (
            str(item).strip()
            and str(item).strip().lower() not in generic_people_terms
            and str(item).strip().lower() in get_available_people()
        )
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
        
    include_count = people_clues.get("include_count")


    if include_count == 1:
        max_score += 5


        person_presence_mask = results["people"].str.strip() != ""


        add_reason(
            results,
            person_presence_mask,
            5,
            "Person present"
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

    st.write(
        "Dog matching diagnostic:",
        results[
            results.apply(
                lambda row: contains_value(
                    row,
                    searchable_columns,
                    "dog"
                ),
                axis=1
            )
        ][
            [
                "photo_id",
                "objects",
                "scene",
                "match_score",
                "match_reasons"
            ]
        ]
    )
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
    candidate_count = len(candidates)
    prompt = f"""
You choose at most one high-value follow-up question for photo retrieval.

Original query:
"{user_query}"

Already extracted clues:
{json.dumps(clues, indent=2)}

Candidate count:
{candidate_count}


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
1. Ask exactly one question when there are more than three candidates.
2. Do not return ask_question false when there are more than three candidates.
3. Ask about a clue that can be applied directly to the candidate metadata.
4. Prefer location, date, object, people, or setting.
5. Do not ask about fine-grained actions such as yawning unless that attribute
   exists in the candidate metadata.
6. Do not ask for a clue the user already provided.
7. For ambiguous locations, ask which location.
8. For indoor or outdoor ambiguity, ask whether the photo was indoors or
   outdoors and set field to setting.
9. Keep the question short and neutral.
10. Return ask_question false only when there are three or fewer candidates.
11. Only ask questions whose answers can be matched against these metadata
    fields:
    location, date, people, objects, scene, description, activities,
    text_visible, and search_terms.

12. Do not ask about breed, size, color, yawning, facial expression, or other
    attributes unless those attributes are explicitly present in the candidate
    etadata.
13. Never ask a follow-up question if its likely answer will not reduce the
    candidate set. Prefer questions about scene, location, date, people, or
    objects that visibly differ among the candidates.
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


    if len(candidates) > 3 and not parsed.get("ask_question"):
        parsed = {
            "ask_question": True,
            "question": (
                "Was the photo taken indoors or outdoors?"
            ),
            "field": "setting",
            "options": [
                "Indoors",
                "Outdoors",
                "Not sure"
            ]
        }


    st.session_state.latest_follow_up_json = parsed

    return parsed

def ask_for_clarification(candidates, user_query, clues):
    return get_follow_up_question(
        candidates,
        user_query,
        clues
    )

# =================================================
# DISPLAY
# =================================================

def show_photo_cards(photo_data, limit=3):
    if photo_data.empty:
        st.warning("No photo matches the selected clues.")
        return


    top_photos = photo_data


    columns = st.columns(min(3, len(top_photos)))


    for index, (_, photo) in enumerate(top_photos.iterrows()):
        if index % 3 == 0:
            columns = st.columns(min(3, len(top_photos) - index))

        with columns[index%3]:
            image_path = SAMPLE_PHOTOS_DIR / str(photo["file_name"])


            if image_path.exists():
                visible_text = format_values(photo["text_visible"])


                tooltip = (
                    f"ID: {photo['photo_id']}\n"
                    f"Date: {photo['date']}\n"
                    f"Location: {photo['location']}\n"
                    f"People: {format_values(photo['people'])}\n"
                    f"Scene: {format_values(photo['scene'])}\n"
                    f"Visible text: {visible_text}"
                )


                st.markdown(
                    f"""
                    <div title="{tooltip}">
                        <img
                            src="data:image/jpeg;base64,{__import__(
                                "base64"
                            ).b64encode(
                                image_path.read_bytes()
                            ).decode()}"
                            width="180"
                            height="180"
                            style="object-fit: cover; border-radius: 14px;"
                        />
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.caption("Hover for details")


                if "confidence" in photo.index:
                    st.metric(
                        "Relative match score",
                        f"{int(photo['confidence'])}%"
                    )
            else:
                st.error(f"Missing image: {photo['file_name']}")

def select_gallery_photo(photo_id):
    st.session_state.gallery_selected_photo = photo_id

@st.fragment
def show_gallery_grid(photo_data):
    if photo_data.empty:
        st.warning("No photos found.")
        return


    selected_photo_id = st.session_state.gallery_selected_photo


    if selected_photo_id:
        selected_photo = photo_data[
            photo_data["photo_id"] == selected_photo_id
        ]


        if not selected_photo.empty:
            photo = selected_photo.iloc[0]
            image_path = SAMPLE_PHOTOS_DIR / str(photo["file_name"])


            card_column, empty_column = st.columns(
                [3, 1],
                gap="small"
            )


            with card_column:
                with st.container(border=True):
                    title_column, close_column = st.columns(
                        [0.85, 0.15]
                    )


                    with title_column:
                        st.subheader(
                            f"Photo details: {photo['photo_id']}"
                        )


                    with close_column:
                        if st.button(
                            "✕",
                            key="close_gallery_photo"
                        ):
                            st.session_state.gallery_selected_photo = None
                            st.rerun(scope="fragment")


                    image_column, details_column = st.columns(
                        [1, 1],
                        gap="large"
                    )


                    with image_column:
                        if image_path.exists():
                            st.image(
                                str(image_path),
                                width=500
                            )
                        else:
                            st.error(
                                f"Missing image: {photo['file_name']}"
                            )


                    with details_column:
                        st.markdown(
                            f"""
                            **Date:**

                            {photo["date"]}

                            **Location:**

                            {photo["location"]}

                            **People:**

                            {format_values(photo["people"])}

                            **Scene:**

                            {format_values(photo["scene"])}

                            **Description:**

                            {photo["description"]}

                            **Objects:**

                            {format_values(photo["objects"])}

                            **Activities:**

                            {format_values(photo["activities"])}

                            **Visible text:**

                            {format_values(photo["text_visible"])}
                            """
                        )


            st.divider()


    for start_index in range(0, len(photo_data), 5):
        row = photo_data.iloc[start_index:start_index + 5]
        columns = st.columns(5)


        for column, (_, photo) in zip(columns, row.iterrows()):
            with column:
                image_path = SAMPLE_PHOTOS_DIR / str(photo["file_name"])


                if image_path.exists():
                    st.image(
                        str(image_path),
                        width=150
                    )


                    st.button(
                        "View details",
                        key=f"gallery_{photo['photo_id']}",
                        on_click=select_gallery_photo,
                        args=(photo["photo_id"],)
                    )
                else:
                    st.error(
                        f"Missing image: {photo['file_name']}"
                    )

def reset_search_state():
    st.session_state.initial_candidates = None
    st.session_state.selected_collection = None
    st.session_state.candidate_groups = None
    st.session_state.follow_up = None
    st.session_state.refined_results = None
    st.session_state.original_memory = ""
    st.session_state.latest_query_json = None
    st.session_state.latest_follow_up_json = None
    st.session_state.clarification_round = 0
    st.session_state.clarification_history = []


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

    if st.session_state.clarification_history:
        st.subheader("Clarification history")


        for index, item in enumerate(
            st.session_state.clarification_history,
            start=1
        ):
            candidates_before = item["candidates_before"]
            candidates_after = item["candidates_after"]


            if candidates_after is None:
                result_text = "Waiting for answer"
            else:
                result_text = (
                    f"{candidates_before} → "
                    f"{candidates_after} candidates"
                )


            st.write(
                f"{index}. {item['question']} "
                f"Answer: {item['answer'] or 'Not answered'} "
                f"({result_text})"
            )

    candidates = st.session_state.initial_candidates


    if candidates is not None:
        st.write("Candidate count:", len(candidates))


    # Generate a follow-up only once for the current candidate set.
    if (
        candidates is not None
        and len(candidates) > 3
        and st.session_state.follow_up is None
    ):
        with st.spinner("Finding the best way to narrow the results..."):
            st.session_state.follow_up = (
                ask_for_clarification(
                    candidates,
                    st.session_state.original_memory,
                    st.session_state.latest_query_json
                )
            )


    # Always render the existing follow-up on every rerun.
    follow_up = st.session_state.follow_up


    if (
        candidates is not None
        and len(candidates) > 3
        and follow_up
        and follow_up.get("ask_question")
    ):
        st.subheader("One more detail could help")
        st.write(follow_up.get("question", ""))


        options = follow_up.get("options", [])


        if not options:
            options = ["Not sure"]


        answer = st.selectbox(
            "Choose an answer",
            ["Not sure"] + options,
            key="current_clarification_answer"
        )
        
        current_query = st.session_state.latest_query_json


        if (
            answer != "Not sure"
            and clue_already_present(answer, current_query)
        ):
            st.info(
                f'"{answer}" is already part of the current search. '
                "Choose a different clue."
            )


        if (
            st.button(
                "Narrow results",
                key="current_narrow_results",
                type="primary"
            )
            and not (
                answer != "Not sure"
                and clue_already_present(
                    answer,
                    st.session_state.latest_query_json
                )
            )
        ):
            try:
                previous_count = len(candidates)


                if answer == "Not sure":
                    refined_candidates = candidates.copy()
                    refined_clues = (
                        st.session_state.latest_query_json
                    )
                    combined_query = (
                        st.session_state.original_memory
                    )
                else:
                    combined_query = (
                        f"{st.session_state.original_memory}. "
                        f"Clarification: {answer}"
                    )


                    with st.spinner("Narrowing your results..."):
                        refined_clues = extract_query_clues(
                            combined_query
                        )


                        clarification_filtered = filter_by_clarification(
                            candidates,
                            answer
                        )


                        if len(clarification_filtered) < len(candidates):
                            refined_candidates = clarification_filtered
                        else:
                            refined_candidates = score_candidates(
                                candidates,
                                refined_clues
                            )


                after_count = len(refined_candidates)


                st.session_state.clarification_history.append({
                    "question": follow_up.get("question", ""),
                    "answer": answer,
                    "candidates_before": previous_count,
                    "candidates_after": after_count
                })


                st.session_state.initial_candidates = (
                    refined_candidates
                )
                st.session_state.latest_query_json = refined_clues
                st.session_state.original_memory = combined_query
                st.session_state.follow_up = None
                st.session_state.latest_follow_up_json = None
                st.session_state.clarification_round += 1


                st.rerun()


            except Exception as error:
                st.error(
                    f"Could not narrow the results: {error}"
                )
    

    
    if candidates is not None and not (
        len(candidates) > 3
        and st.session_state.follow_up
        and st.session_state.follow_up.get("ask_question")
    ):
        if candidates.empty:
            st.warning(
                "I couldn't find a strong match from that memory."
            )


            st.write(
                "One more clue could help narrow it down."
            )


            no_result_options = [
                "Where it was taken",
                "Who was in it",
                "Whether it was indoors or outdoors",
                "Another object or visual detail",
                "Any text visible in the photo",
                "Not sure"
            ]


            no_result_answer = st.selectbox(
                "What kind of clue can you add?",
                no_result_options,
                key="no_result_clue"
            )


            additional_clue = st.text_input(
                "Enter the additional clue",
                placeholder=(
                    "Example: Bengaluru, Rahul, outdoor, beach, medicine..."
                ),
                key="additional_clue"
            )


            if st.button(
                "Try another clue",
                key="no_result_follow_up"
            ):
                if not additional_clue.strip():
                    st.warning("Please enter an additional clue.")
                else:
                    combined_query = (
                        f"{st.session_state.original_memory}. "
                        f"Additional clue type: {no_result_answer}. "
                        f"Additional clue: {additional_clue}"
                    )

                    


                    with st.spinner("Searching again with the new clue..."):
                        try:
                            new_clues = extract_query_clues(
                                combined_query
                            )
                            new_candidates = score_candidates(
                                photos,
                                new_clues
                            )


                            st.session_state.latest_query_json = new_clues
                            st.session_state.initial_candidates = new_candidates
                            st.session_state.original_memory = combined_query
                            st.session_state.selected_collection = None
                            st.session_state.refined_results = None
                            st.session_state.follow_up = None


                            st.rerun()


                        except Exception as error:
                            st.error(
                                f"Could not run the updated search: {error}"
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


                show_photo_cards(candidates)


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
                ["Not sure"] + options,
                key=(
                    f"follow_up_answer_"
                    f"{st.session_state.clarification_round}"
                )
            )


            if st.button(
                "Apply follow-up",
                key=(
                    f"apply_follow_up_"
                    f"{st.session_state.clarification_round}"
                )
            ):
                previous_count = len(selected_collection)
                refined = selected_collection.copy()


                field = follow_up.get("field")
                normalized_answer = answer.strip().lower()


                if normalized_answer in [
                    "indoor",
                    "indoors",
                    "inside",
                    "interior",
                    "outdoor",
                    "outdoors",
                    "outside",
                    "exterior"
                ]:
                    field = "setting"


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
                    if normalized_answer in [
                        "indoor",
                        "indoors",
                        "inside",
                        "interior"
                    ]:
                        setting_values = [
                            "indoor",
                            "indoors",
                            "inside",
                            "interior"
                        ]
                    elif normalized_answer in [
                        "outdoor",
                        "outdoors",
                        "outside",
                        "exterior"
                    ]:
                        setting_values = [
                            "outdoor",
                            "outdoors",
                            "outside",
                            "exterior"
                        ]
                    else:
                        setting_values = [normalized_answer]


                    refined = refined[
                        refined.apply(
                            lambda row: any(
                                value in str(
                                    row.get("scene", "")
                                ).strip().lower()
                                or value in str(
                                    row.get("description", "")
                                ).strip().lower()
                                for value in setting_values
                            ),
                            axis=1
                        )
                    ]


                after_count = len(refined)


                if after_count == previous_count:
                    st.warning(
                        "That answer did not narrow the current results. "
                        "Please choose a different clue."
                    )
                else:
                    st.session_state.clarification_history.append({
                        "question": follow_up["question"],
                        "answer": answer,
                        "candidates_before": previous_count,
                        "candidates_after": after_count
                    })


                    st.session_state.initial_candidates = refined
                    st.session_state.selected_collection = refined
                    st.session_state.refined_results = refined
                    st.session_state.follow_up = None
                    st.session_state.latest_follow_up_json = None
                    st.session_state.clarification_round += 1


                    st.rerun()


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
        "Search photos. NOTE: View Details shows a card on top of the page.",
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

    show_gallery_grid(filtered)


# =================================================
# METADATA PAGE
# =================================================

elif page == "View metadata":
    reload_column, info_column = st.columns([1,4])

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

            metrics[0].metric(
                "Prompt tokens",
                int(logs_dataframe["prompt_tokens"].sum())
            )

            metrics[1].metric(
                "Output tokens",
                int(logs_dataframe["output_tokens"].sum())
            )

            metrics[2].metric(
                "Thinking tokens",
                int(logs_dataframe["thinking_tokens"].sum())
            )

            metrics[3].metric(
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
