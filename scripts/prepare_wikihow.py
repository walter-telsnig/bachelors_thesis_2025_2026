import os
import random
import re
from datasets import load_dataset

# Configuration
OUTPUT_DIR = os.path.join("docs", "wikihow")
NUM_TEXTS = 10

def clean_filename(text):
    """
    Creates a safe filename from the headline.
    Removes invalid characters and truncates length.
    """
    # Remove invalid file characters
    clean = re.sub(r'[\\/*?:"<>|]', "", text)
    # Replace newlines/tabs with space
    clean = clean.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    # Compress whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:60] # Limit length

def prepare_data():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    print("Loading WikiHow dataset from Hugging Face...")
    # Use streaming=True to avoid downloading the entire dataset (gigabytes)
    try:
        dataset = load_dataset("wikihow", "all", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading 'wikihow': {e}")
        print("Trying 'gursi26/wikihow-cleaned' as fallback...")
        try:
             dataset = load_dataset("gursi26/wikihow-cleaned", split="train", streaming=True)
        except Exception as e2:
             print(f"Error loading fallback: {e2}")
             return

    print(f"Selecting {NUM_TEXTS} random texts...")
    # Shuffle with a buffer to get random samples from the stream
    # buffer_size needs to be large enough to provide good randomness
    shuffled_dataset = dataset.shuffle(seed=random.randint(0, 10000), buffer_size=2000)
    
    # Take more than needed in case of filtering
    candidates = shuffled_dataset.take(NUM_TEXTS * 2) 
    
    count = 0
    for item in candidates:
        if count >= NUM_TEXTS:
            break
        
        # Debug structure
        if count == 0:
            print(f"DEBUG: Item keys: {item.keys()}")
            
        headline = item.get('headline', '')
        # Fallback for 'title' if 'headline' is missing (common in some datasets)
        if not headline:
             headline = item.get('title', '')
             
        text = item.get('text', '')
        # Fallback for 'article' if 'text' is missing
        if not text:
             text = item.get('article', '')
        
        if not headline or not text:
            print("Skipping item: Missing headline or text")
            continue
            
        # Clean headline for filename
        filename_base = clean_filename(headline)
        if not filename_base:
            continue
            
        # Define paths
        # Article -> headline.txt
        article_path = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")
        # Summary -> headline_REF.txt (Gold Standard)
        summary_path = os.path.join(OUTPUT_DIR, f"{filename_base}_REF.txt")
        
        # Write files
        try:
            with open(article_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(headline)
                
            print(f"Saved [{count+1}/{NUM_TEXTS}]: {filename_base}")
            count += 1
        except Exception as e:
            print(f"Failed to save {filename_base}: {e}")

    print("Done.")

if __name__ == "__main__":
    prepare_data()
