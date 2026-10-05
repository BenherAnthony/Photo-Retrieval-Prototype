import os
import json
import time
import random
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
# FILE PATHS AND GEMINI CONFIGURATION
# =================================================

BASE_DIR = Path(__file__).parent
SAMPLE_PHOTOS_DIR = BASE_DIR / "Sample Photos"
CSV_PATH = SAMPLE_PHOTOS_DIR / "photos.csv"

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
        "for local use or in Streamlit Cloud Secrets for deployment."
    )
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# =================================================
# LOAD CSV
# =================================================

def load_photos():
    if not CSV_PATH.exists():
        st.error(f"photos.csv was not found at: {CSV_PATH}")
        st.stop()

    return pd.read_csv(CSV_PATH).fillna("")


photos = load_photos()

required_columns = [
    "photo_id",
    "file_name",
    "date_taken",
    "location",
    "people",
    "scene_labels",
    "ocr_text",
    "album_or_event"
]

missing_columns = [
    column
    for column in required_columns
    if column not in photos.columns
]

if missing_columns:
    st.error(
        "Your photos.csv is missing these columns: "
        + ", ".join(missing_columns)
    )
    st.write("Columns currently in photos.csv:", list(photos.columns))
    st.stop()


# =================================================
# FORMATTING HELPERS
# =================================================

def format_pipe_values(value):
    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    return ", ".join(
        item.strip()
        for item in value.split("|")
        if item.strip()
    )


def format_match_reasons(value):
    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    reasons = [
        item.strip().rstrip(";").strip()
        for item in value.split(";")
        if item.strip().rstrip(";").strip()
    ]

    return ", ".join(reasons)


def clean_json_response(response_text):
    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1)

    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1)

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def get_available_people():
    people = set()

    for people_value in photos["people"]:
        for person in str(people_value).split("|"):
            person = person.strip()

            if person:
                people.add(person)

    return sorted(people)


def get_available_locations():
    locations = set()

    for location in photos["location"]:
        location = str(location).strip()

        if location:
            locations.add(location)

    return sorted(locations)


# =================================================
# GEMINI API CALL AND TOKEN LOGGING
# =================================================

def call_gemini_json(prompt, call_type):
    max_attempts = 4
    base_delay_seconds = 2

    for attempt in range(max_attempts):
        try:
            request_started_at = time.strftime("%Y-%m-%d %H:%M:%S")

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            cleaned_response = clean_json_response(response.text)
            parsed_response = json.loads(cleaned_response)

            usage = response.usage_metadata

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
                "model": "gemini-3.8-flash",
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
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "call_type": call_type,
                    "model": "gemini-3.8-flash",
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
# AI STEP 1: EXTRACT USER MEMORY CLUES
# =================================================

def extract_memory_clues(user_memory):
    available_people = get_available_people()
    available_locations = get_available_locations()

    prompt = f"""
You are an AI assistant for a photo-retrieval prototype.

The user is trying to find one old photo using incomplete memories.
Extract only clues explicitly stated or reasonably implied by the user.
Do not invent facts.

User memory:
"{user_memory}"

Available person labels:
{available_people}

Available locations:
{available_locations}

Return ONLY valid JSON. Do not include Markdown or explanation.

Use exactly this JSON structure:

{{
  "location": "",
  "people": [],
  "scene_labels": [],
  "ocr_search_terms": [],
  "objects": []
}}

Rules:

- Use a person only when it appears in the available person-label list.

- Use a location only when it appears in the available location list.

- Break visual descriptions into separate simple labels.

- Example:
  User: "I am looking for a blue building."
  Return:
  "scene_labels": ["blue", "building"]

- Do not return combined labels such as:
  "blue building", "small cafe", "outdoor restaurant".

- Example:
  User: "I want a small cafe photo with food on the table."
  Return:
  "scene_labels": ["cafe", "food", "table"]

- Use "ocr_search_terms" for text the user remembers seeing in the photo.
  Examples include cafe names, restaurant names, medicine names, menu text,
  ticket text, receipt text, shop signs, screenshot text, and banners.

- Example:
  User: "I am looking for a cafe called Sosas."
  Return:
  "scene_labels": ["cafe"],
  "ocr_search_terms": ["Sosas"]

- Example:
  User: "I am looking for Happy Birthday photos."
  Return:
  "scene_labels": ["birthday"],
  "ocr_search_terms": ["Happy Birthday"]

- Do not treat a cafe or shop name as a location unless the user explicitly
  provides a location.

- If there is no valid clue for a field, return an empty string or empty list.
"""

    memory_clues = call_gemini_json(
        prompt=prompt,
        call_type="Memory clue extraction"
    )

    st.session_state.latest_memory_json = memory_clues

    return memory_clues


# =================================================
# INITIAL PHOTO SEARCH: WEIGHTED MATCHING
# =================================================

def get_initial_candidates(photo_data, clues):
    results = photo_data.copy()

    results["match_score"] = 0
    results["match_reasons"] = ""

    location = str(clues.get("location", "")).strip()
    people = clues.get("people", [])
    scene_labels = clues.get("scene_labels", [])
    objects = clues.get("objects", [])
    ocr_search_terms = clues.get("ocr_search_terms", [])

    # ---------------------------------------------
    # Location matches
    # ---------------------------------------------

    if location:
        location_match = results["location"].str.contains(
            location,
            case=False,
            na=False,
            regex=False
        )

        results.loc[location_match, "match_score"] += 5

        results.loc[location_match, "match_reasons"] += (
            f"Location: {location};"
        )

    # ---------------------------------------------
    # People matches
    # ---------------------------------------------

    for person in people:
        people_match = results["people"].str.contains(
            person,
            case=False,
            na=False,
            regex=False
        )

        results.loc[people_match, "match_score"] += 5

        results.loc[people_match, "match_reasons"] += (
            f"Person: {person};"
        )

    # ---------------------------------------------
    # Scene / object labels
    # ---------------------------------------------

    for label in scene_labels + objects:
        label_match = results["scene_labels"].str.contains(
            label,
            case=False,
            na=False,
            regex=False
        )

        results.loc[label_match, "match_score"] += 3

        results.loc[label_match, "match_reasons"] += (
            f"Label: {label};"
        )

    # ---------------------------------------------
    # OCR / remembered text matches
    # ---------------------------------------------

    for text_term in ocr_search_terms:
        ocr_match = results["ocr_text"].str.contains(
            text_term,
            case=False,
            na=False,
            regex=False
        )

        scene_text_match = results["scene_labels"].str.contains(
            text_term,
            case=False,
            na=False,
            regex=False
        )

        event_match = results["album_or_event"].str.contains(
            text_term,
            case=False,
            na=False,
            regex=False
        )

        text_match = (
            ocr_match
            | scene_text_match
            | event_match
        )

        results.loc[text_match, "match_score"] += 4

        results.loc[text_match, "match_reasons"] += (
            f"Text: {text_term};"
        )

    results = results[
        results["match_score"] > 0
    ]

    results = results.sort_values(
        by="match_score",
        ascending=False
    )

    return results


# =================================================
# AI STEP 2: SELECT USEFUL FOLLOW-UP QUESTION
# =================================================

def get_follow_up_question(collection_photos, user_memory):
    candidate_summary = []

    for _, photo in collection_photos.iterrows():
        candidate_summary.append({
            "photo_id": str(photo["photo_id"]),
            "date_taken": str(photo["date_taken"]),
            "people": str(photo["people"]),
            "scene_labels": str(photo["scene_labels"]),
            "ocr_text": str(photo["ocr_text"]),
            "album_or_event": str(photo["album_or_event"])
        })

    prompt = f"""
You are helping a user identify one intended photo from a small collection.

Original user memory:
"{user_memory}"

The user has already selected a candidate collection.
Do NOT ask about location.

Candidate photos:
{json.dumps(candidate_summary, indent=2)}

Return ONLY valid JSON. Do not include Markdown or explanation.

Use exactly this structure:

{{
  "ask_question": true,
  "question": "",
  "field": "people|setting|object|event|none",
  "reason": ""
}}

Rules:

- Ask at most one question.

- Ask a question only if it can clearly reduce the remaining candidate set.

- Use only one of these fields:
  people, setting, object, event, none.

- If no useful question exists, return:
  "ask_question": false
  "question": ""
  "field": "none"

- For people, ask who was in the photo.

- For setting, ask whether it was indoors or outdoors.

- For object, ask about an object or visible wording only if candidates differ.

- For event, ask about an event type only if event labels differ.

- Keep the question short and neutral.

Examples:

If candidate photos differ by people:
"Do you remember who was in the photo?"

If candidate photos differ by setting:
"Do you remember whether this was indoors or outdoors?"

If candidate photos differ by event:
"Was this from a trip or a local outing?"
"""

    follow_up_question = call_gemini_json(
        prompt=prompt,
        call_type="Follow-up question selection"
    )

    st.session_state.latest_follow_up_json = follow_up_question

    return follow_up_question


# =================================================
# REFINE COLLECTION USING FOLLOW-UP ANSWER
# =================================================

def refine_collection(collection_photos, question_field, answer):
    if not answer or answer == "Not sure":
        return collection_photos

    results = collection_photos.copy()

    if question_field == "people":
        results = results[
            results["people"].str.contains(
                answer,
                case=False,
                na=False,
                regex=False
            )
        ]

    elif question_field == "setting":
        results = results[
            results["scene_labels"].str.contains(
                answer,
                case=False,
                na=False,
                regex=False
            )
        ]

    elif question_field == "object":
        results = results[
            results["scene_labels"].str.contains(
                answer,
                case=False,
                na=False,
                regex=False
            )
            |
            results["ocr_text"].str.contains(
                answer,
                case=False,
                na=False,
                regex=False
            )
        ]

    elif question_field == "event":
        results = results[
            results["album_or_event"].str.contains(
                answer,
                case=False,
                na=False,
                regex=False
            )
        ]

    return results


# =================================================
# DISPLAY PHOTO RESULTS
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
                st.error(
                    f"Missing image: {photo['file_name']}"
                )

        with details_column:
            people_display = format_pipe_values(photo["people"])
            labels_display = format_pipe_values(photo["scene_labels"])

            st.markdown(
                f"""
                <div class="photo-details">
                    <h3>{photo["photo_id"]}</h3>
                    <p><strong>Date:</strong> {photo["date_taken"]}</p>
                    <p><strong>Location:</strong> {photo["location"]}</p>
                    <p><strong>People:</strong> {people_display}</p>
                    <p><strong>Labels:</strong> {labels_display}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            ocr_text = str(photo["ocr_text"]).strip()

            if ocr_text:
                st.info(f"Captured text: {ocr_text}")

            event_name = str(photo["album_or_event"]).strip()

            if event_name:
                st.caption(f"Event: {event_name}")

            if "match_score" in photo.index:
                st.metric(
                    "Match strength",
                    int(photo["match_score"])
                )

            if "match_reasons" in photo.index:
                reasons_display = format_match_reasons(
                    photo["match_reasons"]
                )

                if reasons_display:
                    st.markdown(
                        f"""
                        <div class="match-text">
                            Matched because: {reasons_display}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.divider()


# =================================================
# CREATE LOCATION-BASED COLLECTIONS
# =================================================

def get_collection_options(candidate_photos):
    grouped = candidate_photos.groupby(
        "location",
        dropna=False
    )

    options = {}

    for location, group in grouped:
        location_name = str(location).strip()

        if not location_name:
            location_name = "Unknown location"

        collection_name = (
            f"{location_name} — {len(group)} matching photo(s)"
        )

        options[collection_name] = group.copy()

    return options


# =================================================
# SESSION STATE
# =================================================

if "initial_candidates" not in st.session_state:
    st.session_state.initial_candidates = None

if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None

if "follow_up" not in st.session_state:
    st.session_state.follow_up = None

if "refined_results" not in st.session_state:
    st.session_state.refined_results = None

if "original_memory" not in st.session_state:
    st.session_state.original_memory = ""

if "ai_logs" not in st.session_state:
    st.session_state.ai_logs = []

if "latest_memory_json" not in st.session_state:
    st.session_state.latest_memory_json = None

if "latest_follow_up_json" not in st.session_state:
    st.session_state.latest_follow_up_json = None


# =================================================
# HEADER
# =================================================

st.markdown(
    '<div class="main-title">🔎 Vague Memory Photo Retrieval</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Find old photos using partial memories, visual clues,
        people, locations, and text you remember seeing.
    </div>
    """,
    unsafe_allow_html=True
)


# =================================================
# NAVIGATION
# =================================================

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
# PAGE 1: AI-GUIDED PHOTO RETRIEVAL
# =================================================

if page == "Search photos":
    st.subheader("Describe the photo you remember")

    user_memory = st.text_area(
        "What do you remember?",
        placeholder=(
            "Example: I remember a blue building, but I am not sure "
            "whether it was in Bengaluru or Goa."
        ),
        height=120
    )

    if st.button("Search my photos", type="primary"):
        if not user_memory.strip():
            st.warning("Please describe what you remember.")

        else:
            with st.spinner("Finding possible photo collections..."):
                try:
                    clues = extract_memory_clues(user_memory)

                    st.session_state.initial_candidates = (
                        get_initial_candidates(photos, clues)
                    )

                    st.session_state.selected_collection = None
                    st.session_state.follow_up = None
                    st.session_state.refined_results = None
                    st.session_state.original_memory = user_memory

                except Exception as error:
                    error_message = str(error)

                    if "503" in error_message or "UNAVAILABLE" in error_message:
                        st.warning(
                            "The AI service is temporarily busy. "
                            "Please wait a minute and try again."
                        )

                    elif "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
                        st.warning(
                            "The Gemini API quota has been reached. "
                            "Use the sample gallery or wait for the quota reset."
                        )

                    else:
                        st.error(f"Search failed: {error_message}")

    candidates = st.session_state.initial_candidates

    if candidates is not None:
        if candidates.empty:
            st.warning(
                "No photos matched the details remembered so far. "
                "Try another clue, such as people, location, colour, "
                "text, object, or event."
            )

        else:
            st.markdown(
                '<div class="collection-title">Possible photo collections</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Choose the collection that looks most related to the "
                "photo you remember."
            )

            collection_options = get_collection_options(candidates)

            selected_collection_name = st.radio(
                "Candidate collections",
                list(collection_options.keys())
            )

            selected_collection = collection_options[
                selected_collection_name
            ]

            st.markdown(
                f'<div class="collection-title">'
                f'{selected_collection_name}'
                f'</div>',
                unsafe_allow_html=True
            )

            show_photo_cards(selected_collection)

            if st.button("Use this collection", type="primary"):
                st.session_state.selected_collection = selected_collection
                st.session_state.refined_results = None

                if len(selected_collection) > 1:
                    with st.spinner(
                        "Finding the most useful next question..."
                    ):
                        try:
                            st.session_state.follow_up = (
                                get_follow_up_question(
                                    selected_collection,
                                    st.session_state.original_memory
                                )
                            )

                        except Exception:
                            st.session_state.follow_up = {
                                "ask_question": False,
                                "question": "",
                                "field": "none",
                                "reason": ""
                            }

                            st.warning(
                                "The collection is ready, but the AI could "
                                "not create a follow-up question."
                            )

                else:
                    st.session_state.follow_up = {
                        "ask_question": False,
                        "question": "",
                        "field": "none",
                        "reason": ""
                    }

    selected_collection = st.session_state.selected_collection
    follow_up = st.session_state.follow_up

    if selected_collection is not None:
        st.divider()

        current_results = selected_collection

        if (
            follow_up
            and follow_up.get("ask_question")
            and follow_up.get("field") != "none"
            and len(selected_collection) > 1
        ):
            st.subheader("One more detail")

            st.write(follow_up["question"])

            question_field = follow_up["field"]

            if question_field == "people":
                answer_options = ["Not sure"] + get_available_people()

            elif question_field == "setting":
                answer_options = [
                    "Not sure",
                    "Outdoor",
                    "Indoor"
                ]

            elif question_field == "object":
                answer_options = [
                    "Not sure",
                    "Food",
                    "Medicine",
                    "Receipt",
                    "Document",
                    "Building",
                    "Vehicle",
                    "Cake"
                ]

            elif question_field == "event":
                events = sorted(
                    value
                    for value in selected_collection[
                        "album_or_event"
                    ].unique()
                    if str(value).strip()
                )

                answer_options = ["Not sure"] + events

            else:
                answer_options = ["Not sure"]

            answer = st.selectbox(
                "Your answer",
                answer_options
            )

            if st.button("Refine photos"):
                st.session_state.refined_results = refine_collection(
                    selected_collection,
                    question_field,
                    answer
                )

        if st.session_state.refined_results is not None:
            current_results = st.session_state.refined_results

            st.subheader("Refined photo results")
            show_photo_cards(current_results)

        st.divider()

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
# PAGE 2: BASIC KEYWORD SEARCH BASELINE
# =================================================

elif page == "Browse sample gallery":
    st.subheader("Browse the sample photo library")

    st.write(
        "This is the non-AI baseline experience: a basic keyword search "
        "over the same sample photo dataset."
    )

    keyword = st.text_input(
        "Search photos",
        placeholder="Try blue, building, Goa, Sosas, medicine, birthday..."
    ).strip().lower()

    filtered = photos.copy()

    if keyword:
        searchable_columns = [
            "date_taken",
            "location",
            "people",
            "scene_labels",
            "ocr_text",
            "album_or_event"
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
# PAGE 3: CSV METADATA VIEW
# =================================================

elif page == "View metadata":
    reload_column, info_column = st.columns([1, 4])

    with reload_column:
        if st.button("Reload CSV"):
            st.rerun()

    with info_column:
        csv_last_updated = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(CSV_PATH.stat().st_mtime)
        )

        st.caption(f"CSV last modified: {csv_last_updated}")

    st.subheader("Sample photo metadata")

    st.write(
        "This is the manually indexed photo dataset used by the prototype."
    )

    st.dataframe(
        photos,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Debug information")

    st.write("Sample Photos folder:", str(SAMPLE_PHOTOS_DIR))
    st.write("Folder exists:", SAMPLE_PHOTOS_DIR.exists())
    st.write("CSV path:", str(CSV_PATH))
    st.write("CSV exists:", CSV_PATH.exists())

    if not photos.empty:
        first_file_name = str(photos.iloc[0]["file_name"])
        first_image_path = SAMPLE_PHOTOS_DIR / first_file_name

        st.write("First filename from CSV:", first_file_name)
        st.write("Expected image path:", str(first_image_path))
        st.write("First image exists:", first_image_path.exists())


# =================================================
# PAGE 4: AI JSON AND TOKEN LOGS
# =================================================

elif page == "AI JSON & Logs":
    st.subheader("AI JSON & Token Logs")

    memory_tab, follow_up_tab, token_tab = st.tabs([
        "Memory JSON",
        "Follow-up JSON",
        "Token Logs"
    ])

    with memory_tab:
        st.subheader("Latest memory-clue extraction")

        if st.session_state.latest_memory_json:
            st.json(st.session_state.latest_memory_json)
        else:
            st.info(
                "No memory JSON exists yet. Run a search from "
                "'Search photos' first."
            )

    with follow_up_tab:
        st.subheader("Latest follow-up-question JSON")

        if st.session_state.latest_follow_up_json:
            st.json(st.session_state.latest_follow_up_json)
        else:
            st.info(
                "No follow-up JSON exists yet. Select a collection and "
                "click 'Use this collection' first."
            )

    with token_tab:
        st.subheader("Gemini API usage")

        if st.session_state.ai_logs:
            logs_dataframe = pd.DataFrame(
                st.session_state.ai_logs
            )

            total_prompt_tokens = logs_dataframe[
                "prompt_tokens"
            ].sum()

            total_output_tokens = logs_dataframe[
                "output_tokens"
            ].sum()

            total_thinking_tokens = logs_dataframe[
                "thinking_tokens"
            ].sum()

            total_tokens = logs_dataframe[
                "total_tokens"
            ].sum()

            metric_1, metric_2, metric_3, metric_4 = st.columns(4)

            metric_1.metric(
                "Prompt tokens",
                int(total_prompt_tokens)
            )

            metric_2.metric(
                "Output tokens",
                int(total_output_tokens)
            )

            metric_3.metric(
                "Thinking tokens",
                int(total_thinking_tokens)
            )

            metric_4.metric(
                "Total tokens",
                int(total_tokens)
            )

            st.dataframe(
                logs_dataframe,
                use_container_width=True,
                hide_index=True
            )

            if st.button("Clear AI logs"):
                st.session_state.ai_logs = []
                st.session_state.latest_memory_json = None
                st.session_state.latest_follow_up_json = None

                st.rerun()

        else:
            st.info(
                "No Gemini API calls have been made in this browser session."
            )