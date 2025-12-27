import streamlit as st
import requests
import pandas as pd
import nltk
from rouge_score import rouge_scorer
from bert_score import score
import torch

import os
from datetime import datetime
import base64
from bs4 import BeautifulSoup

# --- NLTK Data Check ---
# Ensure 'punkt' and 'punkt_tab' are available for tokenization.
# Uses a fallback to a /tmp directory if standard paths are not writable.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except Exception as e:
        # Fallback to local directory if permission denied (common in some container setups)
        nltk.download('punkt', download_dir='/tmp/nltk_data')
        nltk.download('punkt_tab', download_dir='/tmp/nltk_data')
        nltk.data.path.append('/tmp/nltk_data')

# --- Configuration ---
# Internal Docker network alias for the Flask API
API_URL = "http://summarizer-api:5000"

st.set_page_config(page_title="Thesis Summarization App", layout="wide")

# col_logo is small (1), col_title is large (5)
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

st.title("Comparative Text Summarization")

# CSS to position the logo in the top right corner
logo_path = "icon/logo.png"
logo_base64 = get_base64_image(logo_path)

if logo_base64:
    st.markdown(
        f"""
        <style>
            [data-testid="stHeader"] {{
                background-color: rgba(0,0,0,0);
            }}
            .logo-img {{
                position: fixed;
                top: 2.5rem;
                right: 6rem;
                width: 300px;
                z-index: 99999;
                pointer-events: none;
            }}
        </style>
        <img class="logo-img" src="data:image/png;base64,{logo_base64}">
        """,
        unsafe_allow_html=True
    )

# --- Session State Initialization ---
# We use st.session_state to persist data (input text, summaries) across 
# Streamlit's reactive re-runs (triggered by any button click or interaction).
if 'source_text' not in st.session_state:
    st.session_state['source_text'] = ""
if 'gemini_summary' not in st.session_state:
    st.session_state['gemini_summary'] = ""
if 'classic_summary' not in st.session_state:
    st.session_state['classic_summary'] = ""

# --- Tabs Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Document Input", "Abstractive (Gemini)", "Extractive (Classic)", "Comparative Analysis", "Batch Evaluation"])

# ==========================================
# Tab 1: Document Input & Preprocessing
# ==========================================
with tab1:
    st.header("Document Input")
    
    # User selects how they want to provide the text
    input_method = st.radio("Choose Input Method", ["Manual Text Entry", "Pre-defined File (docs/)", "Upload Text File", "Web URL"])
    
    # 'current_text_content' acts as a buffer for the text currently being viewed/edited in the UI
    # before it is "Processed" and committed to 'source_text'.
    if 'current_text_content' not in st.session_state:
        st.session_state['current_text_content'] = ""

    text_to_process = "" # Local variable to hold the text for the current logic frame

    # --- Method 1: Manual Entry ---
    if input_method == "Manual Text Entry":
        text_to_process = st.text_area("Enter Source Text", height=300, key="manual_input")
        st.session_state['current_text_content'] = text_to_process # Sync with session state
        
    # --- Method 2: Pre-defined File ---
    # Lists .txt files from the /app/docs directory (mounted volume)
    elif input_method == "Pre-defined File (docs/)":
        docs_dir = "/app/docs"
        if os.path.exists(docs_dir):
            # Scan for files in subdirectories (debug/ and wikihow/)
            file_options = []
            
            for root, dirs, files in os.walk(docs_dir):
                for file in files:
                    if file.endswith(".txt") and not file.endswith("_REF.txt"):
                        # Create relative path for display (e.g., "wikihow/cat.txt")
                        rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                        # Ensure we use forward slashes for consistency
                        rel_path = rel_path.replace("\\", "/")
                        file_options.append(rel_path)
            
            # Sort for better UX
            file_options.sort()
            
            st.info(f"Debug: Found {len(file_options)} available documents in /docs")
            
            if file_options:
                selected_rel_path = st.selectbox("Select a file", file_options)
                if selected_rel_path:
                    try:
                        filepath = os.path.join(docs_dir, selected_rel_path)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            text_to_process = file_content
                            st.session_state['current_text_content'] = file_content
                        st.text_area("Preview", value=text_to_process, height=200, disabled=True)
                        st.write(f"Debug: Read {len(text_to_process)} chars from {selected_rel_path}")
                        
                        # Attempt to auto-load reference summary if it exists
                        # (name_REF.txt in same directory)
                        try:
                            base_path = os.path.splitext(filepath)[0]
                            ref_path = f"{base_path}_REF.txt"
                            if os.path.exists(ref_path):
                                with open(ref_path, 'r', encoding='utf-8') as f:
                                        ref_text = f.read()
                                        # You might want to store this in session_state to use in Tab 4 automatically
                                        # st.toast("Gold Reference Summary found and loaded!", icon="ℹ️")
                                        # For now, we can perhaps display it or hint at it, strictly we aren't using it yet 
                                        # directly in Tab 4 unless we add logic there.
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            else:
                st.warning("No eligible .txt files found in /docs subfolders.")
        else:
            st.error("Docs directory not found. Ensure volume is mounted correctly.")
            
    # --- Method 3: File Upload ---
    # Standard Streamlit file uploader for .txt files
    elif input_method == "Upload Text File":
        uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
        if uploaded_file is not None:
            try:
                # Seek to start to ensure we read from beginning (idempotency check)
                uploaded_file.seek(0)
                file_content = uploaded_file.read().decode("utf-8")
                text_to_process = file_content
                st.session_state['current_text_content'] = file_content
                st.text_area("Preview", value=text_to_process, height=200, disabled=True)
                st.write(f"Debug: Uploaded file read, length: {len(text_to_process)}")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                
    # --- Method 4: Web URL ---
    # Uses Requests + BeautifulSoup to scrape text
    elif input_method == "Web URL":
        url = st.text_input("Enter URL")
        if url:
             if st.button("Fetch URL Content"):
                try:
                    with st.spinner("Fetching content..."):
                        # Automatic protocol fix
                        if not url.startswith(('http://', 'https://')):
                             if not url.startswith('http'):
                                 url = 'https://' + url
                        
                        # Add User-Agent to mimic browser and avoid bots protection (e.g. Wikipedia 403)
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        response = requests.get(url, headers=headers)
                        response.raise_for_status()
                        
                        # Parse HTML
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Extract text from ps and headers
                        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
                        fetched_text = "\n\n".join([p.get_text() for p in paragraphs])
                        
                        # Update states
                        st.session_state['current_text_content'] = fetched_text
                        st.session_state['temp_url_text'] = fetched_text
                        text_to_process = fetched_text
                except Exception as e:
                    st.error(f"Error fetching URL: {e}")
        
        # Persist URL text between re-runs
        if 'temp_url_text' in st.session_state and input_method == "Web URL":
            text_to_process = st.session_state['temp_url_text']
            st.session_state['current_text_content'] = text_to_process
            st.text_area("Preview", value=text_to_process, height=200, disabled=True)

    # --- Process Button ---
    # Commits the selected text to 'source_text' for global available
    if st.button("Process Document", type="primary"):
        final_text = st.session_state.get('current_text_content', '')
        
        # Debugging info
        if not final_text:
             st.error(f"Debug: final_text is EMPTY. Input method: {input_method}")
             st.write(f"Debug: st.session_state['current_text_content'] = '{st.session_state.get('current_text_content')}'")
        
        if final_text:
            st.session_state['source_text'] = final_text
            st.success("Document stored successfully!")
            # Note: We reset metrics by rerunning or they auto-update below? 
            # They update below because the script runs top->down.
        else:
            st.warning("No text detected. Please input content first.")
    
    # --- Metrics Display ---
    if st.session_state['source_text']:
        st.divider()
        st.subheader("Document Metrics")
        text = st.session_state['source_text']
        try:
            words = nltk.word_tokenize(text)
            sentences = nltk.sent_tokenize(text)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Word Count", len(words))
            col2.metric("Sentence Count", len(sentences))
            col3.metric("Unique Words", len(set(words)))
        except LookupError:
             st.error("Error: NLTK 'punkt' tokenizer data missing. Please check the logs.")
        except Exception as e:
             st.error(f"Error processing text: {e}")

# ==========================================
# Tab 2: Abstractive (Gemini)
# ==========================================
with tab2:
    st.header("Abstractive Summarization (Gemini)")
    
    col_len, col_purp = st.columns(2)
    
    with col_len:
        length_options = ["3 sentences", "5 sentences", "Short paragraph", "High-level overview", "Detailed", "Custom"]
        length_choice = st.selectbox("Desired Length", length_options, index=1) # Default to 5 sentences
        if length_choice == "Custom":
            length_req = st.text_input("Specify Length", value="50 words")
        else:
            length_req = length_choice

    with col_purp:
        purpose_options = ["General summary", "Technical summary", "Executive summary", "Simple explanation", "Academic style", "Custom"]
        purpose_choice = st.selectbox("Purpose/Focus", purpose_options, index=0)
        if purpose_choice == "Custom":
            purpose_req = st.text_input("Specify Purpose", value="Key insights")
        else:
            purpose_req = purpose_choice

    temp_tooltip = "Controls randomness: Lower values (near 0.0) make text more deterministic and focused. Higher values (near 1.0) make text more creative and diverse."
    temp_req = st.slider("Temperature", 0.0, 1.0, 0.5, help=temp_tooltip)
    
    # Visual Color Indicator (Blue -> Red)
    # R increases from 0 to 255, B decreases from 255 to 0
    red_val = int(temp_req * 255)
    blue_val = int((1 - temp_req) * 255)
    color_hex = f"#{red_val:02x}00{blue_val:02x}"
    
    # Description based on value
    if temp_req < 0.3:
        desc = "Deterministic (Focused)"
    elif temp_req < 0.7:
        desc = "Balanced"
    else:
        desc = "Creative (Diverse)"
        
    st.markdown(f"<span style='color:{color_hex}; font-weight:bold; font-size:1.1em;'>● {desc}</span>", unsafe_allow_html=True)
    
    if st.button("Generate Abstractive Summary"):
        if not st.session_state['source_text']:
            st.error("Please enter source text in Tab 1 first.")
        else:
            payload = {
                "text_content": st.session_state['source_text'],
                "length": length_req,
                "purpose": purpose_req,
                "temperature": temp_req
            }
            try:
                with st.spinner("Calling Gemini API..."):
                    # POST request to Flask API
                    response = requests.post(f"{API_URL}/summarize/gemini", json=payload)
                    if response.status_code == 200:
                        summary = response.json().get("summary", "")
                        st.session_state['gemini_summary'] = summary
                        st.success("Summary Generated!")
                    else:
                        st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

    if st.session_state['gemini_summary']:
        st.subheader("Generated Summary")
        st.write(st.session_state['gemini_summary'])

# ==========================================
# Tab 3: Extractive (Classic)
# ==========================================
with tab3:
    st.header("Extractive Summarization (Classic)")
    
    algo_req = st.selectbox("Algorithm", ["Luhn", "TextRank", "LSA", "LexRank", "TF-IDF"])
    sent_count_req = st.number_input("Sentence Count", min_value=1, value=3)
    
    if st.button("Generate Extractive Summary"):
        if not st.session_state['source_text']:
            st.error("Please enter source text in Tab 1 first.")
        else:
            payload = {
                "text_content": st.session_state['source_text'],
                "algorithm": algo_req,
                "sentences_count": sent_count_req
            }
            try:
                with st.spinner("Processing with Sumy..."):
                    # POST request to Flask API
                    response = requests.post(f"{API_URL}/summarize/classic", json=payload)
                    if response.status_code == 200:
                        response_json = response.json()
                        summary = response_json.get("summary", "")
                        st.session_state['classic_summary'] = summary
                        st.session_state['classic_algorithm'] = algo_req
                        
                        # Store analysis data for highlighting (now available for all algorithms)
                        if "analysis" in response_json:
                            st.session_state['classic_analysis'] = response_json["analysis"]
                        else:
                            st.session_state['classic_analysis'] = []
                            
                        st.success("Summary Generated!")
                    else:
                        st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

    if st.session_state['classic_summary']:
        
        # --- Highlighting Section ---
        if 'classic_analysis' in st.session_state and st.session_state['classic_analysis']:
            st.subheader("Source Text with Highlights")
            analysis_data = st.session_state['classic_analysis']
            
            # Construct HTML for highlighting
            html_content = ""
            for item in analysis_data:
                content = item.get("content", "")
                selected = item.get("selected", False)
                
                # Escape HTML in content just in case
                import html
                content = html.escape(content)
                
                if selected:
                     # Green highlight for selected sentences
                     html_content += f'<span style="background-color: #dcf8c6; color: black; padding: 2px 0; border-radius: 3px;">{content}</span> '
                else:
                     html_content += f'{content} '
            
            # Display in a scrollable container
            with st.container(height=300):
                st.markdown(html_content, unsafe_allow_html=True)
                
        st.divider()

        st.subheader("Generated Summary")
        st.write(st.session_state['classic_summary'])
        
        # Display Analysis Table if available (Only for TF-IDF as requested)
        if 'classic_analysis' in st.session_state and st.session_state['classic_analysis']:
             # Get current algo
             current_algo = st.session_state.get('classic_algorithm', 'Classic')
             
             if current_algo == "TF-IDF":
                with st.expander("Detailed Sentence Analysis"):
                    df = pd.DataFrame(st.session_state['classic_analysis'])
                    # Reorder columns for better readability
                    valid_cols = [c for c in ['sentence_id', 'score', 'selected', 'content'] if c in df.columns]
                    if not df.empty and valid_cols:
                        df = df[valid_cols]
                    
                    # Formatting: Score to 4 decimal places
                    
                    # Create a display copy to avoid mutating state
                    display_df = df.copy()
                    
                    # Visual Indicator: Replace boolean with Green Checkmark / Red Cross
                    display_df['selected'] = display_df['selected'].apply(lambda x: '✅' if x else '❌')
                    
                    # Format score
                    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.4f}")

                    # Convert to interactive Dataframe (User prefers this over Markdown)
                    st.dataframe(
                        display_df, 
                        hide_index=True
                    )

# ==========================================
# Tab 4: Comparative Results & Metrics
# ==========================================
with tab4:
    st.header("Comparative Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Abstractive (Gemini)")
        st.write(st.session_state['gemini_summary'] if st.session_state['gemini_summary'] else "Not generated yet.")
        
    with col_b:
        classic_algo = st.session_state.get('classic_algorithm', 'Classic')
        st.subheader(f"Extractive ({classic_algo})")
        st.write(st.session_state['classic_summary'] if st.session_state['classic_summary'] else "Not generated yet.")
        
    st.markdown("---")
    st.subheader("ROUGE (1, 2, L) & BERTScore Evaluation")
    
    st.info("ℹ️ **Note:** Calculating BERTScore (F1) requires downloading a large model on the first run, which may take some time. Initializing... ")
    
    reference_summary = ""
    
    # Selection for Reference Summary
    ref_input_method = st.radio("Reference Source", ["Manual Input", "Select from File"], horizontal=True)
    
    if ref_input_method == "Select from File":
         docs_dir = "/app/docs"
         if os.path.exists(docs_dir):
            ref_files = []
            for root, dirs, files in os.walk(docs_dir):
                for file in files:
                    if file.endswith("_REF.txt"):
                         rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                         rel_path = rel_path.replace("\\", "/")
                         ref_files.append(rel_path)
            
            ref_files.sort()
            
            if ref_files:
                selected_ref = st.selectbox("Select Reference File", ref_files)
                if selected_ref:
                    try:
                        with open(os.path.join(docs_dir, selected_ref), 'r', encoding='utf-8') as f:
                            reference_summary = f.read()
                        st.info("Reference summary loaded.")
                    except Exception as e:
                        st.error(f"Error reading reference file: {e}")
            else:
                st.warning("No reference files (*_REF.txt) found in docs folder.")
         else:
             st.error("Docs directory not found.")
             
    else:
        reference_summary = st.text_area("Enter Reference (Gold Standard) Summary for Comparison", height=150)

    # If loaded from file, show it in a disabled text area or just use it
    if ref_input_method == "Select from File" and reference_summary:
         st.text_area("Reference Content", value=reference_summary, height=150, disabled=True)
    
    if 'show_scores' not in st.session_state:
        st.session_state['show_scores'] = False

    if st.button("Calculate ROUGE Scores"):
        st.session_state['show_scores'] = True
        
    if st.session_state['show_scores']:
        if not reference_summary:
            st.error("Please provide a reference summary.")
        elif not st.session_state['gemini_summary'] and not st.session_state['classic_summary']:
             st.error("Please generate at least one summary to compare.")
        else:
            # Using rouge_scorer for ROUGE-1, ROUGE-2, and ROUGE-L calculation
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # Results Accumulator
            results_text = f"--- Original Text ---\n{st.session_state['source_text']}\n\n"
            
            # Helper to calculate and return scores string
            def calculate_and_format_scores(candidate_summary, reference_summary, model_name):
                output_str = f"--- {model_name} Summary ---\n{candidate_summary}\n\n"
                output_str += f"--- {model_name} Metrics ---\n"
                
                st.markdown(f"### {model_name} Performance")
                
                # 1. ROUGE Scores
                scores = scorer.score(reference_summary, candidate_summary)
                
                rouge_data = []
                output_str += "ROUGE Scores:\n"
                for metric, result in scores.items():
                    rouge_data.append({
                        "Metric": metric,
                        "Precision": f"{result.precision:.4f}",
                        "Recall": f"{result.recall:.4f}",
                        "F1-Score": f"{result.fmeasure:.4f}"
                    })
                    output_str += f"{metric}: P={result.precision:.4f}, R={result.recall:.4f}, F1={result.fmeasure:.4f}\n"

                st.write("**ROUGE Scores:**")
                st.dataframe(pd.DataFrame(rouge_data))
                output_str += "\n"

                # 2. BERTScore
                try:
                    with st.spinner(f"Calculating BERTScore for {model_name}..."):
                        P, R, F1 = score([candidate_summary], [reference_summary], lang="en", verbose=False)
                        bert_data = [{
                            "Metric": "BERTScore",
                            "Precision": f"{P.item():.4f}",
                            "Recall": f"{R.item():.4f}",
                            "F1-Score": f"{F1.item():.4f}"
                        }]
                        st.write("**BERTScore:**")
                        st.dataframe(pd.DataFrame(bert_data))
                        output_str += f"BERTScore: P={P.item():.4f}, R={R.item():.4f}, F1={F1.item():.4f}\n\n"
                except Exception as e:
                    st.error(f"Error calculating BERTScore: {e}")
                    output_str += f"BERTScore Error: {e}\n\n"
                
                return output_str

            # Comparison for Gemini
            if st.session_state['gemini_summary']:
                gemini_res = calculate_and_format_scores(st.session_state['gemini_summary'], reference_summary, "Gemini")
                results_text += gemini_res
                
            # Comparison for Classic
            if st.session_state['classic_summary']:
                classic_algo = st.session_state.get('classic_algorithm', 'Classic')
                classic_res = calculate_and_format_scores(st.session_state['classic_summary'], reference_summary, f"Extractive ({classic_algo})")
                results_text += classic_res
            
            st.divider()
            
            # Formatted Download Filename with Timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"summarization_results_{timestamp}.txt"
            
            st.download_button(
                label="Download Full Results (.txt)",
                data=results_text,
                file_name=file_name,
                mime="text/plain"
            )

# ==========================================
# Tab 5: Batch Evaluation
# ==========================================
with tab5:
    st.header("Batch Summarization & Evaluation")
    
    st.info("Run extractive summarization on a folder of texts and compare them against reference summaries.")
    
    # Helper to find all subdirectories recursively
    def get_all_subdirs(root_dir):
        subdirs = []
        if os.path.exists(root_dir):
            # Add root itself
            subdirs.append(root_dir)
            for root, dirs, files in os.walk(root_dir):
                for d in dirs:
                    # Append absolute path
                    subdirs.append(os.path.join(root, d).replace("\\", "/"))
        return sorted(subdirs)

    docs_root = "/app/docs"
    available_dirs = get_all_subdirs(docs_root)
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        # source_dir = st.text_input("Source Directory", value="/app/docs") 
        if available_dirs:
            source_dir = st.selectbox("Source Directory", available_dirs, index=0, key="batch_source_dir")
        else:
            st.error(f"No directories found in {docs_root}")
            source_dir = docs_root # Fallback

    with col_b2:
        # ref_dir = st.text_input("Reference Directory", value="/app/docs")
        if available_dirs:
            ref_dir = st.selectbox("Reference Directory", available_dirs, index=0, key="batch_ref_dir")
        else:
            ref_dir = docs_root

    col_b3, col_b4 = st.columns(2)
    with col_b3:
        batch_algo = st.selectbox("Algorithm (Extractive)", ["TF-IDF", "Luhn", "TextRank", "LSA", "LexRank"], index=0, key="batch_algo")
    with col_b4:
        batch_len = st.number_input("Summary Length (Sentences)", min_value=1, max_value=20, value=5, step=1, key="batch_len")

    if st.button("Run Batch Evaluation (Extractive + Gemini)"):
        if not os.path.exists(source_dir) or not os.path.exists(ref_dir):
            st.error("Invalid directories provided.")
        else:
            files_processed = []
            
            # 1. Scan Source Files
            source_files = [f for f in os.listdir(source_dir) if f.endswith(".txt") and not f.endswith("_REF.txt")]
            
            if not source_files:
                st.warning("No source files found (excluding *_REF.txt).")
            else:
                 # Initialize Scorer
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                
                # Progress Bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_data = []

                for i, filename in enumerate(source_files):
                    status_text.text(f"Processing {filename}...")
                    
                    # Read Source
                    try:
                        with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as f:
                            source_text = f.read()
                    except Exception as e:
                        st.error(f"Error reading {filename}: {e}")
                        continue

                    # Find Reference
                    ref_filename = filename.replace(".txt", "_REF.txt")
                    ref_path_check = os.path.join(ref_dir, ref_filename)
                    
                    if not os.path.exists(ref_path_check):
                        # Try exact match if filename already contains _REF (unlikely given filter) or typical naming
                        # Fallback: check if exact filename exists in ref dir (if user separated folders)
                        if os.path.exists(os.path.join(ref_dir, filename)):
                             ref_path_check = os.path.join(ref_dir, filename)
                        else:
                            st.warning(f"No reference found for {filename} (Matched: {ref_filename}) - Skipping evaluation.")
                            continue
                    
                    try:
                        with open(ref_path_check, 'r', encoding='utf-8') as f:
                            reference_text = f.read()
                    except Exception as e:
                         st.error(f"Error reading ref {ref_filename}: {e}")
                         continue
                            
                    # Call Extractive API
                    extractive_summary = ""
                    gemini_summary = ""
                    
                    # 1. Extractive
                    try:
                        payload = {
                            "text_content": source_text,
                            "algorithm": batch_algo,
                            "sentences_count": batch_len
                        }
                        response = requests.post(f"{API_URL}/summarize/classic", json=payload)
                        if response.status_code == 200:
                            extractive_summary = response.json().get("summary", "")
                        else:
                            st.error(f"Extractive API Error for {filename}: {response.text}")
                    except Exception as e:
                        st.error(f"Extractive Processing Error {filename}: {e}")

                    # 2. Gemini (Abstractive)
                    try:
                        gemini_payload = {
                            "text_content": source_text,
                            "length": f"{batch_len} sentences",
                            "purpose": "General Summary",
                            "temperature": 0.7
                        }
                        gemini_response = requests.post(f"{API_URL}/summarize/gemini", json=gemini_payload)
                        if gemini_response.status_code == 200:
                            gemini_summary = gemini_response.json().get("summary", "")
                        else:
                             st.error(f"Gemini API Error for {filename}: {gemini_response.text}")
                    except Exception as e:
                         st.error(f"Gemini Processing Error {filename}: {e}")

                    # Calculate Scores
                    result_row = {"File": filename}
                    
                    # Extractive Scores
                    if extractive_summary:
                        r_scores_ex = scorer.score(reference_text, extractive_summary)
                        P_ex, R_ex, F1_ex = score([extractive_summary], [reference_text], lang="en", verbose=False)
                        
                        result_row["Extractive R1"] = round(r_scores_ex['rouge1'].fmeasure, 4)
                        result_row["Extractive R2"] = round(r_scores_ex['rouge2'].fmeasure, 4)
                        result_row["Extractive RL"] = round(r_scores_ex['rougeL'].fmeasure, 4)
                        result_row["Extractive BERT"] = round(F1_ex.item(), 4)
                    else:
                         result_row["Extractive R1"] = 0.0

                    # Gemini Scores
                    if gemini_summary:
                        r_scores_gem = scorer.score(reference_text, gemini_summary)
                        P_gem, R_gem, F1_gem = score([gemini_summary], [reference_text], lang="en", verbose=False)
                        
                        result_row["Gemini R1"] = round(r_scores_gem['rouge1'].fmeasure, 4)
                        result_row["Gemini R2"] = round(r_scores_gem['rouge2'].fmeasure, 4)
                        result_row["Gemini RL"] = round(r_scores_gem['rougeL'].fmeasure, 4)
                        result_row["Gemini BERT"] = round(F1_gem.item(), 4)
                    else:
                        result_row["Gemini R1"] = 0.0

                    results_data.append(result_row)

                    # Update Progress
                    progress_bar.progress((i + 1) / len(source_files))
                
                status_text.text("Batch Processing Complete!")
                
                if results_data:
                    df_results = pd.DataFrame(results_data)
                    
                    st.divider()
                    st.subheader("Aggregated Metrics")
                    
                    # Calculate Averages
                    avg_metrics = df_results.mean(numeric_only=True)
                    st.dataframe(avg_metrics.to_frame(name="Average Score").T)
                    
                    st.subheader("Detailed Results")
                    st.dataframe(df_results)
                    
                    # Download DataFrame
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Batch Results (.csv)",
                        csv,
                        "batch_results.csv",
                        "text/csv",
                        key='batch-download-csv'
                    )
                else:
                    st.warning("No files were successfully processed.")
