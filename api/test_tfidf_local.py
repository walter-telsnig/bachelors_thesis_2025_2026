from summarizer_tfidf import TFIDFSummarizer
import sys

def test_summarizer():
    print("Testing TF-IDF Summarizer...")
    
    text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans or animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems.
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
    """
    
    summarizer = TFIDFSummarizer()
    summary = summarizer(text, 2)
    
    print(f"\nOriginal Sentence Count: {len(text.split('.'))}")
    print(f"Summary Sentence Count: {len(summary)}")
    print("\nGenerated Summary:")
    for s in summary:
        print(f"- {s}")
    
    if len(summary) == 2:
        print("\nSUCCESS: Summary generated with correct length.")
    else:
        print(f"\nFAILURE: Expected 2 sentences, got {len(summary)}.")
        sys.exit(1)

if __name__ == "__main__":
    test_summarizer()
