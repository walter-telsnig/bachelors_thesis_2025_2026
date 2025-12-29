# Project Documentation: Extended README

This document provides a detailed explanation of the project structure, specifically focusing on the `api`, `ui`, and `docker-compose` configurations. This document also serves as a technical companion to Chapter 8 (Implementation) of the Bachelor’s thesis and provides additional details not fully covered in the main text

## Project Overview

This application is a **Comparative Text Summarization Tool** that allows users to generate and evaluate summaries using two different approaches:
1.  **Abstractive Summarization**: Powered by Google's Gemini Pro model (via API).
2.  **Extractive Summarization**: Powered by the `sumy` library (Luhn, TextRank, LSA, LexRank) and a custom TF-IDF implementation.

The application is containerized using Docker and split into a backend API (Flask) and a frontend UI (Streamlit).

---

## 1. Root Configuration

### `docker-compose.yml`
This is the main orchestration file that defines how the application services run together.
-   **Services**:
    -   `summarizer-api`: Builds the backend from the `./api` directory. It runs on port `5000`.
    -   `summarizer-ui`: Builds the frontend from the `./ui` directory. It runs on port `8501` and depends on `summarizer-api`.
-   **Networking**: Creates a bridge network `summarizer-network` to allow the UI to communicate with the API (e.g., the UI calls `http://summarizer-api:5000`).
-   **Volumes**: Mounts the local `./docs` directory to `/app/docs` inside the UI container, allowing the app to access local text files for summarization.
-   **Environment**: Passes the `GEMINI_API_KEY` from the local `.env` file to the API container.

---

## 2. API Directory (`/api`)

The backend service responsible for processing summarization requests.

### `api/app.py`
The entry point for the Flask application. It defines the REST API endpoints used by the frontend.
-   **`POST /summarize/gemini`**:
    -   Accepts `text_content`, `length`, `purpose`, and `temperature`.
    -   Uses the Google GenAI client to generate an abstractive summary based on the provided constraints.
-   **`POST /summarize/classic`**:
    -   Accepts `text_content`, `algorithm` (Luhn, TextRank, LSA, LexRank, TF-IDF), and `sentences_count`.
    -   Uses the `sumy` library or the custom `TFIDFSummarizer` to extract key sentences from the text.
    -   Returns the summary text and "analysis data" (which sentences were selected) for highlighting in the UI.
-   **NLTK Handling**: Automatically checks for and downloads necessary NLTK data (`punkt`) if missing.

### `api/summarizer_tfidf.py`
A custom implementation of an extractive summarizer using TF-IDF (Term Frequency-Inverse Document Frequency).
-   **Logic**:
    1.  Tokenizes text into sentences.
    2.  Calculates the TF-IDF matrix for the sentences.
    3.  Computes a "centroid" vector representing the average of all sentences.
    4.  Ranks sentences based on their Cosine Similarity to the centroid.
    5.  Returns the top $N$ sentences that represent the core meaning of the text.

### `api/Dockerfile`
Defines the environment for the Flask API.
-   Installs Python dependencies (`flask`, `google-genai`, `sumy`, `nltk`, `scikit-learn`, etc.).
-   Exposes port 5000.
-   Runs `app.py` on container startup.

---

## 3. UI Directory (`/ui`)

The frontend service built with Streamlit, providing an interactive interface for the user.

### `ui/app.py`
The main application script driving the user interface. It is organized into 5 tabs:

1.  **Tab 1: Document Input**:
    -   Allows users to input text via:
        -   Manual typing.
        -   Selecting a file from the mounted `docs/` directory (filters out `_REF.txt` files).
        -   Uploading a `.txt` file.
        -   Fetching text from a Web URL (includes basic scraping logic).
    -   Displays document metrics (word count, sentence count).

2.  **Tab 2: Abstractive (Gemini)**:
    -   Controls for Gemini API: Summary Length, Purpose/Focus, and Temperature (creativity).
    -   Sends a request to `summarizer-api/summarize/gemini`.
    -   Displays the generated summary.

3.  **Tab 3: Extractive (Classic)**:
    -   Controls for Algorithm selection (Luhn, TextRank, etc.) and sentence count.
    -   Sends a request to `summarizer-api/summarize/classic`.
    -   **Highlighting**: Displays the original source text with the selected summary sentences highlighted in green.
    -   **Analysis Table**: Specifically for TF-IDF, shows a detailed table of sentence scores.

4.  **Tab 4: Comparative Analysis**:
    -   Compare side-by-side results of Gemini vs. Extractive summaries.
    -   **Evaluation**: Calculates ROUGE (1, 2, L) and BERTScore metrics.
    -   Requires a "Reference" (Gold Standard) summary, which can be loaded from a file or entered manually.

5.  **Tab 5: Batch Evaluation**:
    -   Automates the evaluation process for an entire folder of documents.
    -   Iterates through source files, matches them with reference files (`_REF.txt`), generates summaries using both methods, and compiles a CSV report of performance metrics (ROUGE, BERTScore).

### `ui/Dockerfile`
Defines the environment for the Streamlit UI.
-   Installs heavy dependencies like `torch`, `transformers` (for BERTScore), `bullet` (rendering), etc.
-   Exposes port 8501.
-   Runs `streamlit run app.py` on container startup.

### `ui/.streamlit/`
Contains configuration for the Streamlit server (e.g., theme, server settings).

---

## 4. Other Key Files

### `.env`
Stores sensitive configuration, primarily the `GEMINI_API_KEY`. This file is **not** committed to version control.

### `docs/`
A local directory meant to store test documents. It is mounted into the UI container so users can easily select files for processing.
