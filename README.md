# Google Photos Vague-Memory Retrieval Prototype

An AI-powered prototype that helps users retrieve photos from a personal collection when they remember the memory but do not know the exact date, location, filename, or searchable keywords.

## Live App

Add your deployed Streamlit URL here:

[Open the live prototype](PASTE_YOUR_STREAMLIT_URL_HERE)

## Repository Purpose

This repository contains the MVP for a Google Photos retrieval experience focused on vaguely remembered photos.

The prototype explores how a user can search for a photo using natural-language memory, such as:

- “Show me pictures of a dog indoors.”
- “Find the photo of the medicine I took when I was sick.”
- “Show me the café from my Goa trip.”
- “Find the picture with a dog but no people.”

The system interprets the user’s description, searches a photo metadata index, presents candidate photos, and asks a follow-up question when several candidates remain.

## Problem

Traditional photo search works best when users know precise attributes such as:

- Date.
- Location.
- Person’s name.
- Album.
- Filename.
- Object visible in the image.
- Text contained in the image.

However, people often remember a photo as a vague visual memory rather than as searchable metadata.

For example:

> “I remember a picture of a dog inside the house, but I do not remember when I took it.”

The prototype investigates whether conversational clarification can help users narrow down these ambiguous retrieval attempts.

## Product Hypothesis

If users can describe a vague visual memory in natural language and answer one or more targeted clarification questions, then they should be able to identify the intended photo more successfully than by relying only on exact keyword search.

## Core User Flow

1. The user describes the photo they are trying to find.
2. The system extracts structured clues from the description.
3. The system searches the indexed photo collection.
4. Candidate photos are displayed.
5. If multiple candidates remain, the system asks a follow-up question.
6. The user selects an additional clue.
7. The candidate set is narrowed.
8. The user reviews the remaining photos and selects the relevant result.

## Example

Initial query:

```text
Show me pictures of a dog indoors.
```

The system may identify:

- Object: dog.
- Scene: indoor.
- People constraint: not specified.
- Location: not specified.
- Date: not specified.

If several photos match, the system may ask:

```text
Are there people in the photo?
```

After the user answers, the system updates the candidate set and displays the remaining results.

## Features

- Natural-language photo retrieval.
- Structured extraction of search clues.
- Object and scene-based matching.
- People inclusion and exclusion constraints.
- Candidate photo display.
- Clarification questions for ambiguous results.
- Candidate count tracking.
- Clarification history.
- Support for vague descriptions rather than exact filenames or dates.

## Technical Stack

- Python.
- Streamlit.
- Pandas.
- Google Gemini API.
- JSON-based photo metadata.
- GitHub.
- Streamlit Community Cloud.

## Project Structure

```text
photo-retrieval-prototype/
│
├── app.py
├── generated_metadata.json
├── requirements.txt
├── README.md
└── data/
    └── sample photo files or metadata
```

Update this structure if your repository contains additional folders or files.

## How It Works

### 1. Query understanding

The user enters a natural-language description. The application sends the description to the language model, which extracts structured clues such as:

```json
{
  "location_mentions": [],
  "date_clues": [],
  "people_clues": {
    "include_people": [],
    "exclude_people": [],
    "has_people_constraint": false
  },
  "object_clues": {
    "include": ["dog"],
    "exclude": []
  },
  "activity_clues": {
    "include": [],
    "exclude": []
  },
  "free_text_clues": {
    "include": ["indoor"],
    "exclude": []
  },
  "needs_clarification": false
}
```

### 2. Candidate retrieval

The extracted clues are compared against the indexed photo metadata. Candidate photos are ranked according to how closely their metadata matches the query.

### 3. Clarification

If the result set remains ambiguous, the application generates a follow-up question.

The answer is then applied to the current candidate set instead of restarting from the full photo collection.

### 4. Result refinement

The system records the candidate transition after each clarification, for example:

```text
7 → 3 candidates
```

This helps evaluate whether a clarification question meaningfully improves retrieval.

## Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd photo-retrieval-prototype
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the API key

Create a `.env` file locally:

```text
GOOGLE_API_KEY=your_api_key_here
```

Do not commit `.env` to GitHub.

For Streamlit Community Cloud, add the API key through the application’s Secrets settings rather than placing it directly in the repository.

### 5. Run the application

```bash
streamlit run app.py
```

The application should open in your browser at:

```text
http://localhost:8501
```

## Environment Variables

The application requires:

```text
GOOGLE_API_KEY
```

If your code uses a different variable name, update this section to match the implementation.

## MVP Scope

This is a functional product prototype rather than a production-ready Google Photos feature.

The MVP focuses on:

- Vague-memory retrieval.
- Natural-language query understanding.
- Metadata-based candidate matching.
- Conversational clarification.
- Candidate reduction.
- User evaluation of retrieved photos.

## Current Limitations

- The prototype uses a limited sample photo collection.
- Photo metadata may be manually generated or simulated.
- Search quality depends on metadata completeness.
- Some visual attributes may not be represented in the metadata.
- Clarification questions may not always reduce the candidate set.
- The prototype does not yet connect to a user’s real Google Photos library.
- The system does not represent a final production ranking model.
- User research and retrieval success need to be evaluated with more participants.
- The system may interpret similar concepts inconsistently.
- The API key and model behavior can affect results.

## Evaluation Metrics

Potential metrics for evaluating the prototype include:

### Primary metric

Percentage of vague-memory retrieval attempts where the user identifies the intended photo.

```text
Successful retrieval rate =
Successful retrieval attempts / Total vague-memory retrieval attempts
```

### Supporting metrics

- Candidate reduction after clarification.
- Number of clarification questions per successful retrieval.
- Time to identify the intended photo.
- Percentage of searches ending without a recognized result.
- Percentage of clarification questions that reduce the candidate set.
- User-reported confidence in the retrieved result.
- Number of abandoned retrieval attempts.

## Research Context

This prototype is part of a product case study exploring:

- How users remember old visual information.
- Which attributes users remember or forget.
- How users describe vague memories.
- Where photo retrieval breaks down.
- Whether clarification can improve successful retrieval.

The goal is not to improve generic photo search. The focus is specifically on retrieving photos that users remember but cannot precisely describe.

## Privacy and Security

This repository should not contain:

- API keys.
- `.env` files.
- Private personal photos.
- Personally identifiable information.
- Private Google Photos exports.
- Unredacted user conversations.

Use synthetic, sample, or permissioned data for demonstration and research.

## Future Improvements

Possible future directions include:

- Direct integration with Google Photos.
- Visual embedding-based retrieval.
- Better scene and setting classification.
- Automatic detection of redundant clarification questions.
- Information-gain-based question selection.
- Personalized memory timelines.
- Search over OCR text and documents.
- Better handling of uncertainty.
- User feedback on candidate relevance.
- Evaluation with real users and larger photo collections.
- Explanation of why each candidate was retrieved.

## Status

This project is an MVP prototype for product discovery and evaluation.

It is intended to test the hypothesis that conversational clarification can help users retrieve vaguely remembered photos from a personal photo collection.

## Author

Add your name here.

## License

Add a license here if required.
