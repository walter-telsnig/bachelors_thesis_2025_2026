from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

class TFIDFSummarizer:
    def __init__(self):
        # Ensure punkt is available (should be handled by app.py, but good for safety)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def __call__(self, text, sentences_count):
        return self.summarize(text, sentences_count)

    def summarize(self, text, sentences_count):
        """
        Summarizes the text using TF-IDF Centroid Similarity.
        
        Args:
            text (str): The input text.
            sentences_count (int): Number of sentences to return.
            
        Returns:
            tuple: (list of selected sentences, list of analysis dicts)
        """
        if not text:
            return [], []

        # 1. Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Prepare analysis structure if text is too short
        if len(sentences) <= sentences_count:
            analysis = []
            for i, sent in enumerate(sentences):
                analysis.append({
                    "sentence_id": i + 1,
                    "score": 1.0, # Dummy score
                    "selected": True,
                    "content": sent
                })
            return sentences, analysis

        # 2. Vectorize sentences (TF-IDF)
        # using 'english' stop words for better filtering
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
             # Handle empty vocabulary or too short text
             analysis = []
             for i, sent in enumerate(sentences):
                # Fallback: Select first k
                selected = i < sentences_count
                analysis.append({
                    "sentence_id": i + 1,
                    "score": 0.0,
                    "selected": selected,
                    "content": sent
                })
             return sentences[:sentences_count], analysis

        # 3. Compute Centroid (Mean of all sentence vectors)
        # axis=0 means across rows (sentences)
        centroid_vector = np.asarray(tfidf_matrix.mean(axis=0))

        # 4. Compute Cosine Similarity between each sentence and the centroid
        # cosine_similarity expects 2D arrays, so we convert centroid to array if needed
        # centroid_vector is matrix, shape (1, n_features)
        similarities = cosine_similarity(tfidf_matrix, centroid_vector)
        
        # Flatten to 1D array of scores
        scores = similarities.flatten()

        # 5. Select Top-K sentences
        # argsort returns indices of elements sorted in ascending order
        # we take last 'sentences_count' (highest scores)
        top_indices = scores.argsort()[-sentences_count:]
        
        # Sort indices to preserve original order
        top_indices_set = set(top_indices)
        top_indices_sorted = sorted(top_indices)
        
        final_sentences = [sentences[i] for i in top_indices_sorted]
        
        # 6. Build Analysis Data
        analysis = []
        for i, sent in enumerate(sentences):
            analysis.append({
                "sentence_id": i + 1,
                "score": float(scores[i]),
                "selected": i in top_indices_set,
                "content": sent
            })
        
        return final_sentences, analysis
