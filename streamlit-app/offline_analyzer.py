import re
import json
import os
import fitz  # PyMuPDF
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import (
    list_pdfs,
    extract_text_from_pdf_gcs,
    generate_gemini_content,
    store_feedback_bq,
    get_access_token, # Needed for some setups, good practice to have
)

# --- Configuration ---
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Main Logic ---

def parse_sentiment_response(sentiment_text):
    """
    Parses the model's response to extract sentiment and rationale for each asset class.
    Expected format per line: "Asset Class: Sentiment | Rationale"
    """
    normalization_map = {
        "Equities": "Equity",
        "Equity": "Equity",
        "Fixed Income": "Fixed Income",
        "Commodities": "Commodities",
        "Commodity": "Commodities",
        "Real Estate": "Real Estate",
        "Crypto": "Crypto",
    }
    # A set of the canonical asset class names for quick lookup
    valid_assets = set(normalization_map.values())
    parsed_data = {}

    # Regex to capture Asset Class, Sentiment, and an optional Rationale
    pattern = re.compile(r"^\s*([^:]+?)\s*:\s*(Buy|Sell|Hold)\s*(?:\|\s*(.*))?$", re.I | re.MULTILINE)

    for match in pattern.finditer(sentiment_text):
        asset, sentiment, rationale = match.groups()
        normalized_asset = normalization_map.get(asset.strip())

        if normalized_asset:
            parsed_data[normalized_asset] = {
                "sentiment": sentiment.strip().capitalize(),
                "rationale": rationale.strip() if rationale else "No rationale provided.",
            }

    # Ensure all expected asset classes have an entry, even if not found in the response
    for asset in valid_assets:
        if asset not in parsed_data:
            parsed_data[asset] = {"sentiment": "Hold", "rationale": "No sentiment provided by the model."}

    return parsed_data

def get_prediction(text):
    """Generates the sentiment prediction prompt and calls the Gemini API."""
    prompt = f"""You are a financial analyst. Based on the following text, give a sentiment (Buy, Sell, or Hold) and a brief one-sentence rationale for each asset class: "Equity", "Fixed Income", "Commodities", "Real Estate", "Crypto".

Respond with each asset class on a new line, in the format 'Asset Class: Sentiment | Rationale'. For example:
Equity: Buy | Positive earnings outlook and strong market momentum.
Fixed Income: Hold | Yields are stable but there is potential for volatility.
Text:
{text}"""
    # Use the centralized Gemini function
    return generate_gemini_content(prompt)

def extract_date_from_pdf(pdf_bytes: bytes) -> str | None:
    """
    Extracts a date from the first page of a PDF using regex.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) > 0:
            first_page_text = doc[0].get_text("text")
            # Regex to find dates like "Month Day, Year" or "MM/DD/YYYY" etc.
            date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            match = re.search(date_pattern, first_page_text, re.IGNORECASE)
            if match:
                return match.group(0)
    except Exception as e:
        print(f"Could not extract date: {e}")
    return None

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Uploads a local file to a GCS bucket."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

def build_and_save_vector_store(chunks, metadata, gcs_bucket_name):
    """Builds and saves the FAISS index and metadata locally and to GCS."""
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Encoding {len(chunks)} text chunks into embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    data_to_store = [{"text": chunk, "metadata": meta} for chunk, meta in zip(chunks, metadata)]

    # Save locally first
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

    print(f"Saving FAISS index locally to {index_path}")
    faiss.write_index(index, index_path)

    print(f"Saving metadata locally to {metadata_path}")
    with open(metadata_path, "wb") as f:
        pickle.dump(data_to_store, f)

    print("Local vector store built successfully.")

    # Upload to GCS
    gcs_prefix = "CIOInsights/vectors"
    print(f"\nUploading vector store to gs://{gcs_bucket_name}/{gcs_prefix}...")
    upload_to_gcs(index_path, gcs_bucket_name, f"{gcs_prefix}/index.faiss")
    upload_to_gcs(metadata_path, gcs_bucket_name, f"{gcs_prefix}/metadata.pkl")
    print("Vector store uploaded to GCS successfully.")

def run_batch_analysis():
    """
    Lists all PDFs in the GCS bucket, runs sentiment analysis on each,
    and stores the results in BigQuery.
    """
    if not all([BUCKET_NAME, PROJECT_ID, DATASET_ID, TABLE_ID]):
        print("Error: One or more environment variables are not set.")
        print("Please ensure BUCKET_NAME, PROJECT_ID, DATASET_ID, and TABLE_ID are exported.")
        return

    print(f"Starting batch analysis for PDFs in gs://{BUCKET_NAME}...")
    pdf_files = list_pdfs(BUCKET_NAME)

    if not pdf_files:
        print("No PDF files found in the bucket.")
        return

    all_chunks = []
    all_metadata = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
    )

    for pdf_file in pdf_files:
        print(f"\n--- Analyzing: {pdf_file} ---")
        try:
            # This function already filters headers, footers, and disclaimers
            text, dt = extract_text_from_pdf_gcs(BUCKET_NAME, pdf_file)
            if not text:
                print(f"Warning: No text could be extracted from {pdf_file}. Skipping.")
                continue

            # --- 1. Sentiment Analysis ---
            print("Running sentiment analysis...")
            prediction_text = get_prediction(text)
            parsed_data = parse_sentiment_response(prediction_text)
            predicted_labels = {k: v['sentiment'] for k, v in parsed_data.items()}

            row_to_insert = {
                "document": pdf_file, "extracted_text": text,
                "prediction_text": prediction_text,
                "parsed_sentiment": json.dumps(parsed_data),
                "predicted_labels": json.dumps(predicted_labels),
                "is_different": False, "USER": "batch"
            }

            errors = store_feedback_bq(PROJECT_ID, DATASET_ID, TABLE_ID, row_to_insert)
            if errors:
                print(f"Error storing feedback for {pdf_file}: {errors}")
            else:
                print(f"Successfully analyzed and stored results for {pdf_file}.")

            # --- 2. RAG Data Preparation ---
            print("Preparing data for RAG index...")
            # We need the raw bytes to extract the date, which isn't returned
            # by the text extraction function. We'll re-download for this.
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(pdf_file)
            pdf_bytes = blob.download_as_bytes()
            
            publication_date = dt
            print(f"Extracted Date: {publication_date}")
            
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": pdf_file,
                    "date": publication_date
                })
            print(f"Processed {len(chunks)} chunks for RAG from {pdf_file}.")

        except Exception as e:
            print(f"An unexpected error occurred while processing {pdf_file}: {e}")
            # Print more details of the exception for debugging
            import traceback
            traceback.print_exc()

    # --- 3. Build and Save Vector Store ---
    if all_chunks:
        print("\n--- Building FAISS index from all processed documents ---")
        build_and_save_vector_store(all_chunks, all_metadata, BUCKET_NAME)

    print("\n--- Batch analysis complete. ---")

if __name__ == "__main__":
    run_batch_analysis()