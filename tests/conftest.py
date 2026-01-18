import pytest
import sys
import os
from unittest.mock import MagicMock

# --- Global Mocks for Heavy/External Dependencies ---
# We mock these BEFORE importing app to prevent invalid imports or downloads (like NLTK)

# 1. Mock NLTK
mock_nltk = MagicMock()
mock_nltk.data.find.return_value = True # Pretend data exists
sys.modules['nltk'] = mock_nltk

# 2. Mock Sumy and submodules
mock_sumy = MagicMock()
sys.modules['sumy'] = mock_sumy
sys.modules['sumy.parsers.plaintext'] = MagicMock()
sys.modules['sumy.nlp.tokenizers'] = MagicMock()
sys.modules['sumy.summarizers.luhn'] = MagicMock()
sys.modules['sumy.summarizers.text_rank'] = MagicMock()
sys.modules['sumy.summarizers.lsa'] = MagicMock()
sys.modules['sumy.summarizers.lex_rank'] = MagicMock()

# 3. Mock Google GenAI
mock_google = MagicMock()
mock_genai = MagicMock()
mock_google.genai = mock_genai
sys.modules['google'] = mock_google
sys.modules['google.genai'] = mock_genai
sys.modules['google.genai.types'] = MagicMock()

# 4. Mock TFIDF Summarizer (custom module that might import nltk)
sys.modules['summarizer_tfidf'] = MagicMock()

# --- Path Setup & App Import ---
# Add the 'api' directory to sys.path
# tests/conftest.py -> ../api
api_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../api'))
if api_path not in sys.path:
    sys.path.insert(0, api_path)

# Now we can import the app. 
# Because of the sys.modules hacks above, 'import nltk' inside app.py will return our mock.
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
