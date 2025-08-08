from google.cloud import storage, bigquery
import streamlit as st
import io
import os
import faiss
import pickle
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import re
from datetime import datetime, timedelta

# --- Configuration ---
SERVICE_ACCOUNT_FILE = "key.json"
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")

# --- Constants for RAG ---
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def list_pdfs_one(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs():
        if blob.name.endswith('.pdf'):
            return [blob.name]  # Return a list with only the first PDF found
    return []

def list_pdfs(bucket_name, n=5):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    pdfs = [blob.name for blob in bucket.list_blobs() if blob.name.endswith('.pdf')]
    return pdfs[:n]

def extract_text_from_pdf_gcs(bucket_name, pdf_blob_name):
    """
    Extracts clean text from a PDF in GCS, filtering out headers, footers, images,
    and all content after the first disclaimer page. Also extracts the date.
    Returns (text, date_str)
    """
    # 1. Download PDF from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(pdf_blob_name)
    pdf_data = blob.download_as_bytes()

    # 2. Open PDF
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
    except Exception as e:
        print(f"Could not open PDF {pdf_blob_name} with PyMuPDF. Error: {e}")
        return "", None

    disclaimer_keywords = ["important information", "disclaimer", "disclosure"]
    content_text = []
    disclaimer_found = False

    # --- Extract date from the first page(s) ---
    date_str = None
    date_pattern = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})")
    for i in range(min(2, len(doc))):  # Check first two pages for a date
        page_text = doc[i].get_text("text")
        match = date_pattern.search(page_text)
        if match:
            date_str = match.group(1)
            break

    # --- Extract content up to the first disclaimer page ---
    for i in range(len(doc)):
        page = doc[i]
        page_text_lower = page.get_text("text").lower()
        if any(keyword in page_text_lower for keyword in disclaimer_keywords):
            disclaimer_found = True
            break  # Stop at the first disclaimer page

        page_height = page.rect.height
        content_rect = fitz.Rect(
            page.rect.x0, page_height * 0.10,
            page.rect.x1, page_height * 0.90
        )
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] == 0:  # Only text blocks
                block_rect = fitz.Rect(block[:4])
                if content_rect.intersects(block_rect):
                    content_text.append(block[4])

    doc.close()
    return "\n".join(content_text), date_str

@st.cache_data(ttl=3600)
def get_access_token() -> str:
    """Gets access token from the service account file."""
    
    service_account_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not service_account_file:
        print("ERROR: The GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")

    if not os.path.exists(service_account_file):
        print(f"ERROR: Service account file not found at '{service_account_file}'.")
        raise FileNotFoundError(f"Service account file not found at '{service_account_file}'.")
    try:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=scopes
        )
        creds.refresh(Request())
        return creds.token
    except Exception as e:
        print(f"ERROR: Failed to get access token from service account file: {e}")
        raise

@st.cache_resource
def generate_gemini_content(prompt: str, model_id: str = "gemini-2.5-pro"):
    """Calls the Gemini API to generate content."""
    # Log the prompt to the console for debugging purposes
    print("\n--- PROMPT SENT TO GEMINI ---\n")
    print(prompt)
    print("\n-----------------------------\n")

    token = get_access_token()
    api_endpoint = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{model_id}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "topP": 0.95}
    }
    
    response = requests.post(api_endpoint, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model, cached for performance."""
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_vector_store():
    """Downloads and loads the FAISS index and metadata from GCS."""
    bucket_name = "hackathon-qga"
    prefix = "CIOInsights/vectors"
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        print(f"Downloading index from gs://{bucket_name}/{prefix}/index.faiss")
        bucket.blob(f"{prefix}/index.faiss").download_to_filename(index_path)

        print(f"Downloading metadata from gs://{bucket_name}/{prefix}/metadata.pkl")
        bucket.blob(f"{prefix}/metadata.pkl").download_to_filename(metadata_path)

        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        st.error(f"Failed to load vector store from GCS: {e}")
        return None, None

def find_relevant_insights(query: str, index, metadata, model, k=5) -> str:
    """Finds relevant text chunks from the vector store using FAISS."""
    if index is None or not metadata:
        return ""
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k)
    
    # metadata is a list of dicts: [{"text": chunk, "metadata": {"source": filename}}]
    results = [metadata[i]["text"] for i in indices[0] if i < len(metadata)]
    return "\n\n---\n\n".join(results)

def find_relevant_insights_batch(queries: list[str], index, metadata, model, k=5, days_filter: int = None) -> list[dict]:
    """
    Finds relevant text chunks for a batch of queries, which is much more
    performant than calling find_relevant_insights for each query individually.
    It can also filter results to include only documents within a specific number of days.

    Args:
        queries (list[str]): A list of query strings.
        index: The FAISS index.
        metadata: The metadata list corresponding to the index.
        model: The sentence-transformer embedding model.
        k (int): The number of nearest neighbors to retrieve.
        days_filter (int, optional): If provided, only include results from documents
                                     published within this many days. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries, each with "context" and "sources" keys.
    """
    if index is None or not metadata:
        return [{"context": "", "sources": []}] * len(queries)

    # Encode all queries in a single batch
    query_embeddings = model.encode(queries, convert_to_numpy=True)

    # If filtering by date, we need to retrieve more results initially to have a good pool for filtering.
    search_k = k * 5 if days_filter is not None else k

    # FAISS search is optimized for batch queries
    distances, indices = index.search(np.array(query_embeddings).astype("float32"), search_k)

    batched_results = []
    cutoff_date = datetime.utcnow() - timedelta(days=days_filter) if days_filter is not None else None

    # Process results for each query
    for i in range(len(queries)):
        query_indices = indices[i]

        texts = []
        sources = set()
        # Filter out invalid indices (-1) and assemble results
        for j in query_indices:
            # Stop once we have enough results for this query
            if len(texts) >= k:
                break

            if j != -1 and j < len(metadata):
                item_data = metadata[j]
                item_metadata = item_data.get("metadata", {})

                # Date filtering logic
                if cutoff_date:
                    item_date_str = item_metadata.get("date")
                    if not item_date_str:
                        continue  # Skip if no date and we are filtering
                    try:
                        # The date format from offline_analyzer is 'Month Day, Year'
                        item_date = datetime.strptime(item_date_str, "%B %d, %Y")
                        if item_date < cutoff_date:
                            continue  # Skip if the document is too old
                    except (ValueError, TypeError):
                        continue # Skip if date is malformed or not a string

                texts.append(item_data["text"])
                source_doc = item_metadata.get("source")
                if source_doc:
                    sources.add(source_doc)

        context_str = "\n\n---\n\n".join(texts)
        batched_results.append({"context": context_str, "sources": sorted(list(sources))})
    return batched_results

def store_feedback_bq(project_id, dataset_id, table_id, row):
    client = bigquery.Client(project=project_id)
    table_ref = client.dataset(dataset_id).table(table_id)
    errors = client.insert_rows_json(table_ref, [row])
    # The insert_rows_json method returns an empty list on success,
    # and a list of errors if any rows failed.
    return errors
