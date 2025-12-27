from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from summarizer_tfidf import TFIDFSummarizer
import nltk

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Verify NLTK data presence at startup.
# We check for both 'punkt' and 'punkt_tab' which are required for sentence tokenization in newer NLTK versions.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # Download if missing (fallback for non-docker environments or build issues)
    nltk.download('punkt')
    nltk.download('punkt_tab')

def get_gemini_client():
    """
    Initializes and returns the Google GenAI client using the API key from environment variables.
    Raises ValueError if GEMINI_API_KEY is missing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)

@app.route('/summarize/gemini', methods=['POST'])
def summarize_gemini():
    """
    Endpoint for Abstractive Summarization using Google Gemini.
    Expects JSON payload: { 'text_content', 'length', 'purpose', 'temperature' }
    Returns JSON: { 'summary': 'generated text' }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        text_content = data.get('text_content', '')
        length = data.get('length', '3 sentences')
        purpose = data.get('purpose', 'general summary')
        temperature = float(data.get('temperature', 0.5))

        if not text_content:
             return jsonify({"error": "text_content is required"}), 400

        client = get_gemini_client()
        
        # System instruction defines the persona
        system_instruction = "You are an expert abstractive summarizer. Your goal is to provide concise, accurate, and coherent summaries based on the user's requirements."
        
        # User prompt structure enforcing specific constraints
        prompt = f"""
        Please provide an abstractive summary of the following text.
        
        Constraints:
        - Length: Approximately {length}.
        - Focus/Purpose: {purpose}.
        - Style: Abstractive (rewrite in your own words, do not just extract sentences).
        
        Source Text:
        {text_content}
        """

        # Call the Gemini API models.generate_content
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=0.95, 
            )
        )

        return jsonify({"summary": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summarize/classic', methods=['POST'])
def summarize_classic():
    """
    Endpoint for Extractive Summarization using Sumy algorithms.
    Expects JSON payload: { 'text_content', 'algorithm', 'sentences_count' }
    Returns JSON: { 'summary': 'concatenated sentences' }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        text_content = data.get('text_content', '')
        algorithm = data.get('algorithm', 'Luhn')
        sentences_count = int(data.get('sentences_count', 5))

        if not text_content:
             return jsonify({"error": "text_content is required"}), 400

        # Initialize Sumy parser with English tokenizer
        parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
        
        # Select the requested summarizer algorithm
        summarizer = None
        if algorithm == "Luhn":
            summarizer = LuhnSummarizer()
        elif algorithm == "TextRank":
            summarizer = TextRankSummarizer()
        elif algorithm == "LSA":
            summarizer = LsaSummarizer() # Requires numpy
        elif algorithm == "LexRank":
            summarizer = LexRankSummarizer() # Requires numpy
        elif algorithm == "TF-IDF":
            # TF-IDF Summarizer (Custom Implementation)
            custom_tfidf = TFIDFSummarizer()
            # Returns tuple (sentences, analysis_data)
            summary_sentences, analysis_data = custom_tfidf(text_content, sentences_count)
            
            return jsonify({
                "summary": " ".join(summary_sentences),
                "analysis": analysis_data
            })
            
        else:
             return jsonify({"error": f"Unsupported algorithm: {algorithm}"}), 400

        # Generate summary (list of sentences) for Sumy algorithms
        summary_sentences = summarizer(parser.document, sentences_count)
        
        # Combine sentence objects into a single string for return
        final_summary = " ".join([str(sentence) for sentence in summary_sentences])

        # Generate analysis data for highlighting
        # We need to map which sentences from the original doc were selected
        analysis_data = []
        
        # Convert summary sentences to strings for easier comparison
        # Note: Sumy sentences are objects, but str(s) gives the text.
        # Direct object comparison works if they are the exact same instances.
        selected_sentences_set = set(summary_sentences) 
        
        for i, sentence in enumerate(parser.document.sentences):
            is_selected = sentence in selected_sentences_set
            analysis_data.append({
                "sentence_id": i,
                "content": str(sentence),
                "selected": is_selected
                # 'score' is specific to algorithms, tricky to get normalized for all, skipping for generic highlight
            })

        return jsonify({
            "summary": final_summary,
            "analysis": analysis_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000, accessible externally (0.0.0.0)
    app.run(host='0.0.0.0', port=5000)
