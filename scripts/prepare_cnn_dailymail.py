import os
import random
import re
from datasets import load_dataset
import hashlib

# Configuration
OUTPUT_DIR = os.path.join("docs", "cnn_dailymail")
NUM_TEXTS = 10

def clean_filename(text):
    """
    Creates a safe filename from the content if no title is available.
    Uses first few words.
    """
    # Remove invalid file characters
    clean = re.sub(r'[\\/*?:"<>|]', "", text)
    # Replace newlines/tabs with space
    clean = clean.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    # Compress whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:50] # Limit length

def prepare_data():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    print("Loading CNN/DailyMail dataset (abisee/cnn_dailymail) from Hugging Face...")
    try:
        # '3.0.0' is the standard configuration
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Selecting {NUM_TEXTS} random texts...")
    shuffled_dataset = dataset.shuffle(seed=random.randint(0, 10000), buffer_size=2000)
    
    candidates = shuffled_dataset.take(NUM_TEXTS * 2)
    
    count = 0
    for item in candidates:
        if count >= NUM_TEXTS:
            break
            
        # CNN/DM fields: 'article', 'highlights', 'id'
        article = item.get('article', '')
        highlights = item.get('highlights', '')
        doc_id = item.get('id', '')
        
        if not article or not highlights:
            continue
            
        # Use ID for filename to be unique, but maybe check if we can generate a readable one
        # Let's use the first few words of the article as a readable filename
        readable_name = clean_filename(article)
        if not readable_name:
            readable_name = doc_id
        else:
             # Append short hash to ensure uniqueness if headers are similar
             short_hash = hashlib.md5(doc_id.encode()).hexdigest()[:6]
             readable_name = f"{readable_name}_{short_hash}"

        filename_base = readable_name
        
        # Define paths
        # Article -> filename.txt
        article_path = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")
        # Summary -> filename_REF.txt (Gold Standard)
        summary_path = os.path.join(OUTPUT_DIR, f"{filename_base}_REF.txt")
        
        # Write files
        try:
            with open(article_path, "w", encoding="utf-8") as f:
                f.write(article)
            
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(highlights)
                
            print(f"Saved [{count+1}/{NUM_TEXTS}]: {filename_base}")
            count += 1
        except Exception as e:
            print(f"Failed to save {filename_base}: {e}")

    print("Done.")

if __name__ == "__main__":
    prepare_data()
