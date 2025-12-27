# Comparative Text Summarization: Abstractive (LLM) vs. Extractive (Classic)

## Project Overview
This project is developed for a Bachelor's Thesis to compare two distinct approaches to automatic text summarization:
1.  **Abstractive Summarization**: Uses Large Language Models (LLMs), specifically the **Gemini 2.5 Flash** model, to understand the source text and rewrite a summary in its own words, similar to how a human would.
2.  **Extractive Summarization**: Uses classic Natural Language Processing (NLP) algorithms (Luhn, TextRank, LSA, LexRank, TF-IDF) via the **Sumy** library to score and extract the most significant sentences from the original text without modification.

The application allows users to input text, run both methods with customizable parameters, and side-by-side compare the results using quantitative metrics like **ROUGE-1, ROUGE-2, ROUGE-L** and **BERTScore**. It also supports **Batch Evaluation** to process entire datasets at once.

## Architecture Diagram (Textual)

The application follows a microservices architecture containerized with Docker:

1.  **User Layer (Frontend)**: 
    -   **Streamlit UI**: A web interface running on port 8501.
    -   Handles user input, parameter selection, and result visualization.
    -   Communicates with the backend via HTTP POST requests.

2.  **Logic Layer (Backend - API)**:
    -   **Flask API**: A RESTful service running on port 5000.
    -   **Endpoint `/summarize/gemini`**: Connects to the external Google Gemini API for abstractive summarization.
    -   **Endpoint `/summarize/classic`**: Runs local NLP algorithms (Sumy/NLTK) for extractive summarization.

3.  **External Layer**:
    -   **Google Gemini API**: Cloud provider for LLM inference.

**Flow**:  
`User Input (Streamlit)` -> `HTTP POST` -> `Flask API` -> (`Gemini Client` OR `Sumy Engine`) -> `Result` -> `Streamlit Display`

## Setup & Deployment

### Prerequisites
-   Docker and Docker Compose installed.
-   A Google Gemini API Key (get it from [Google AI Studio](https://aistudio.google.com/)).

### Step-by-Step Instructions

1.  **Clone/Open Project**: Ensure you are in the project root directory.

2.  **Configure Environment Variables**:
    -   Create a file named `.env` in the root directory.
    -   Add your API key:
        ```env
        GEMINI_API_KEY=your_actual_api_key_here
        ```

3.  **Build and Run**:
    -   Execute the following command to build the images and start the containers:
        ```bash
        docker-compose up --build
        ```
    -   Wait for the logs to indicate that both `summarizer-api` and `summarizer-ui` are running.

4.  **Access the Application**:
    -   Open your web browser and navigate to: [http://localhost:8501](http://localhost:8501)

## Usage Guide

### Tab 1: Document Input & Preprocessing
-   **Goal**: Load the text you want to summarize.
-   **Input Methods**:
    1.  **Manual Text Entry**: Paste text directly into the text area.
    2.  **Pre-defined File**: Select a file from the `docs/` folder (mounted from your project root).
    3.  **Upload Text File**: Upload a `.txt` file from your computer.
    4.  **Web URL**: Enter a URL (e.g., Wikipedia) to scrape text from. The system handles HTML parsing automatically.
-   **Process**: Click "Process Document" to normalize the text and calculate statistics.
-   **Metrics**: View basic stats like Word Count, Sentence Count, and Unique Words.

### Tab 2: Abstractive (Gemini)
-   **Goal**: Generate a summary that rewritten and condenses meaning.
-   **Settings**:
    -   *Length*: Specify constraints (Dropdown: "Short", "Medium", "Long" or Custom "X sentences").
    -   *Purpose*: Guide the model's focus (Dropdown: "General", "Academic", etc. or Custom).
    -   *Temperature*: Adjust creativity (0.0 = Blue/Deterministic, 1.0 = Red/Creative).
-   **Action**: Click "Generate Abstractive Summary".

### Tab 3: Extractive (Classic)
-   **Goal**: Create a summary by picking the "best" sentences from the source.
-   **Settings**:
    -   *Algorithm*: Choose the math behind sentence scoring (Luhn, TextRank, etc.).
    -   *Sentence Count*: Choose exactly how many sentences to keep.
-   **Visualization**: View the original text with **highlighted green sentences**, showing exactly which parts were selected by the algorithm.
-   **Action**: Click "Generate Extractive Summary".

### Tab 4: Comparative Results & Metrics
-   **View**: See the Abstractive and Extractive summaries side-by-side.
-   **Evaluate**:
    -   Paste a "Reference Summary" (Gold Standard) - this is a human-written summary to compare against.
    -   Click "Calculate ROUGE Scores".
    -   Click "Calculate ROUGE Scores".
    -   Review the metrics to gauge how well the machine summaries match the reference:
        -   **ROUGE-1**: Overlap of unigrams (1-grams).
        -   **ROUGE-2**: Overlap of bigrams (2-grams).
        -   **ROUGE-L**: Longest Common Subsequence.
        -   **BERTScore**: Semantic similarity using contextual embeddings.
    -   **Download**: Functionality to save the full report (Source, Summaries, Scores) as a `.txt` file with a timestamp.
    
    > [!WARNING]
    > **BERTScore Usage**: The first time you run BERTScore, the application will download a pre-trained model (approx. 400MB+). This requires an active internet connection and will take some extra time. Subsequent runs will use the cached model and be faster.

### Tab 5: Batch Evaluation
-   **Goal**: Evaluate algorithms on an entire dataset at once.
-   **Inputs**:
    -   *Source Directory*: Select the folder containing input text files (e.g., `docs/cnn_dailymail`).
    -   *Reference Directory*: Select the folder containing gold standard summaries (`_REF.txt`).
    -   *Algorithm*: One extractive algorithm to test against Gemini.
-   **Process**:
    -   Iterates through every source file.
    -   Generates **both** an Extractive and an Abstractive (Gemini) summary.
    -   Compares both against the reference summary.
-   **Output**:
    -   Aggregated Average Scores Table.
    -   Detailed File-by-File Results Table.
    -   **Download**: Export all results as `.csv`.

## Implemented Algorithms

1.  **Gemini 2.5 Flash (Abstractive)**
    -   A modern Large Language Model (LLM) by Google. It understands context, semantics, and can synthesize information into new sentences.

2.  **Luhn (Extractive)**
    -   A heuristic method based on word frequency. Importance is determined by the density of "significant" words (frequent but not stop-words).

3.  **TextRank (Extractive)**
    -   Graph-based algorithm inspired by PageRank. Sentences are nodes, and similarity (shared words) forms edges. Important sentences are "central" in the graph.

4.  **LSA (Latent Semantic Analysis) (Extractive)**
    -   Uses Singular Value Decomposition (SVD) to capture latent relationships between terms and sentences. It identifies semantically important sentences.

5.  **LexRank (Extractive)**
    -   Another graph-based approach similar to TextRank but uses cosine similarity (tf-idf vectors) for connectivity and handles centroid scoring differently.

6.  **TF-IDF (Extractive)**
    -   Term Frequency-Inverse Document Frequency. Scores sentences based on the sum of the TF-IDF scores of their constituent words. Sentences with rare but distinct words are ranked higher.
