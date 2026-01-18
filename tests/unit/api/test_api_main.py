import pytest
import json

def test_health_check(client):
    # The app doesn't have a root health check, but we can try a 404 to ensure app is running
    response = client.get('/')
    assert response.status_code == 404

def test_summarize_gemini_no_data(client):
    response = client.post('/summarize/gemini', json={})
    assert response.status_code == 400
    assert "No data provided" in response.get_json()['error']

def test_summarize_gemini_missing_text(client):
    response = client.post('/summarize/gemini', json={'length': '3 sentences'})
    assert response.status_code == 400
    assert "text_content is required" in response.get_json()['error']

def test_summarize_gemini_success(client):
    # Since we mocked google.genai globally in conftest, we need to inspect the mock to set return values.
    # However, 'from app import app' has already run.
    # We can rely on the fact that app.py calls `client.models.generate_content`.
    # We need to access the mocked client instance inside the app or the mocked library.
    
    # Access the global mock we set in conftest
    import sys
    mock_genai = sys.modules['google.genai']
    
    # Setup the mock return value
    # client = genai.Client() -> mocks.generate_content -> response.text
    mock_response = sys.modules['unittest.mock'].MagicMock()
    mock_response.text = "This is a mocked summary."
    
    # When app calls genai.Client(api_key=...), it returns a mock (let's call it client_instance)
    # client_instance.models.generate_content(...) returns mock_response
    mock_genai.Client.return_value.models.generate_content.return_value = mock_response

    payload = {
        'text_content': 'Some long text to summarize.',
        'length': 'short',
        'purpose': 'testing',
        'temperature': 0.5
    }
    
    # We also need to ensure GEMINI_API_KEY is set, or mocked out in os.environ
    # app.py checks os.getenv("GEMINI_API_KEY")
    # We can mock os.environ or just set it temporarily
    import os
    os.environ["GEMINI_API_KEY"] = "fake_key"

    response = client.post('/summarize/gemini', json=payload)
    
    if response.status_code != 200:
        print(f"DEBUG Error response: {response.get_json()}")

    assert response.status_code == 200
    assert response.get_json()['summary'] == "This is a mocked summary."

def test_summarize_classic_no_data(client):
    response = client.post('/summarize/classic', json={})
    assert response.status_code == 400

def test_summarize_classic_success(client):
    # Access global mocks
    import sys
    from unittest.mock import MagicMock
    
    # Mocking PlaintextParser and Summarizers
    # In app.py: 
    # parser = PlaintextParser.from_string(...)
    # summarizer = LuhnSummarizer()
    # summary_sentences = summarizer(parser.document, ...)
    
    # 1. Setup Parser Mock
    mock_parser_cls = sys.modules['sumy.parsers.plaintext'].PlaintextParser
    mock_doc = MagicMock()
    
    # Setup sentences list for highlighting
    mock_sent1 = MagicMock()
    mock_sent1.__str__.return_value = "Sentence 1."
    mock_doc.sentences = [mock_sent1]
    
    mock_parser_cls.from_string.return_value.document = mock_doc

    # 2. Setup Summarizer Mock (Luhn is default)
    mock_luhn_cls = sys.modules['sumy.summarizers.luhn'].LuhnSummarizer
    mock_summarizer_instance = mock_luhn_cls.return_value
    
    # Result of summarization acts like a list of sentences
    mock_summarizer_instance.return_value = [mock_sent1]

    payload = {
        'text_content': 'Sentence 1. Sentence 2.',
        'algorithm': 'Luhn',
        'sentences_count': 1
    }
    response = client.post('/summarize/classic', json=payload)

    assert response.status_code == 200
    data = response.get_json()
    assert "Sentence 1." in data['summary']
    assert len(data['analysis']) > 0
