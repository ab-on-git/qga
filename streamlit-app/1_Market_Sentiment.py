import streamlit as st
from utils import list_pdfs, extract_text_from_pdf_gcs, store_feedback_bq, generate_gemini_content
import os
import re
import json
from google.cloud import bigquery
from datetime import datetime, timedelta

#st.set_page_config(layout="wide") 
st.set_page_config(page_title="Market Sentiment", layout="wide")

# Constants
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
    prompt = f"""You are a financial analyst. Based on the following text, give a sentiment (Buy, Sell, or Hold) and a brief four-sentence rationale for each asset class: "Equity", "Fixed Income", "Commodities", "Real Estate", "Crypto".

Respond with each asset class on a new line, in the format 'Asset Class: Sentiment | Rationale'. For example:
Equity: Buy | Positive earnings outlook and strong market momentum.
Fixed Income: Hold | Yields are stable but there is potential for volatility.

Text:
{text[:20_000]}"""
    # Use the centralized Gemini function
    return generate_gemini_content(prompt)

# Streamlit UI
st.title("CIO Insight on Market Sentiments")

# Initialize session state to hold analysis results across reruns
if "sentiment" not in st.session_state:
    st.session_state.sentiment = None
if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = None
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "video_operation_name" not in st.session_state:
    st.session_state.video_operation_name = None
if "video_url" not in st.session_state:
    st.session_state.video_url = None
if "video_error" not in st.session_state:
    st.session_state.video_error = None

#pdfs = list_pdfs(BUCKET_NAME)
#selected_pdf = st.selectbox("Select a document", pdfs)

range_options = {
    "1 month": 30,
    "2 months": 60,
    "3 months": 90,
    "6 months": 180,
    "1 year": 365,
}
selected_range = st.selectbox("Select document date range for analysis", list(range_options.keys()))

def get_documents_in_range(project_id, dataset_id, table_id, days):
    client = bigquery.Client(project=project_id)
    cutoff_date = (datetime.utcnow() - timedelta(days=days)).date()
    query = f"""
        SELECT document, document_dt
        FROM `{project_id}.{dataset_id}.{table_id}`
    """
    df = client.query(query).to_dataframe()
    def parse_dt(dt_str):
        try:
            return datetime.strptime(dt_str, "%B %d, %Y").date()
        except Exception:
            return None
    df["parsed_dt"] = df["document_dt"].apply(parse_dt)
    filtered = df[df["parsed_dt"] >= cutoff_date]
    return filtered["document"].dropna().unique().tolist()

days = range_options[selected_range]

doc_names = get_documents_in_range(PROJECT_ID, DATASET_ID, TABLE_ID, days)
if doc_names:
    with st.expander(f"**{len(doc_names)} documents found in selected range**", expanded=True):
        # Create a markdown string with a bulleted list of documents
        doc_list_md = ""
        for doc in doc_names:
            doc_list_md += f"- `{doc}`\n"
        st.markdown(doc_list_md)
else:
    st.info("No documents found in the selected date range.")

def get_feedback_texts_in_range(project_id, dataset_id, table_id, days):
    client = bigquery.Client(project=project_id)
    cutoff_date = (datetime.utcnow() - timedelta(days=days)).date()
    # BigQuery expects date in 'YYYY-MM-DD', but your column is 'May 25, 2024'
    # So we need to parse and compare as strings in Python
    query = f"""
        SELECT extracted_text, document_dt
        FROM `{project_id}.{dataset_id}.{table_id}`
    """

    #print(query)
    df = client.query(query).to_dataframe()
    # Parse document_dt and filter in Python
    def parse_dt(dt_str):
        try:
            return datetime.strptime(dt_str, "%B %d, %Y").date()
        except Exception:
            return None
    df["parsed_dt"] = df["document_dt"].apply(parse_dt)
    filtered = df[df["parsed_dt"] >= cutoff_date]
    return " ".join(filtered["extracted_text"].dropna().tolist())

if st.button("Analyze"):
    days = range_options[selected_range]
    #st.session_state.selected_pdf = selected_pdf
    with st.spinner("Extracting text and getting sentiment..."):
        #text = extract_text_from_pdf_gcs(BUCKET_NAME, st.session_state.selected_pdf)
        all_text = get_feedback_texts_in_range(PROJECT_ID, DATASET_ID, TABLE_ID, days)
        # Store the extracted text in the session state
        st.session_state.document_text = all_text
        # Use rerun to update the UI and enter the feedback section below
        st.session_state.sentiment = get_prediction(all_text)
        st.rerun()

# This block will now run after "Analyze" is clicked and the state is set
if st.session_state.sentiment:
    st.subheader("Sentiment Analysis Results")

    # Parse LLM response to get sentiment and rationale
    sentiment_data = parse_sentiment_response(st.session_state.sentiment)

    asset_classes = ["Equity", "Fixed Income", "Commodities", "Real Estate", "Crypto"]
    options = ["Buy", "Sell", "Hold"]

    with st.container(border=True):
        st.markdown("#### Model Sentiments & Rationales")
        sentiment_colors = {
            "Buy": "#d4edda",    # light green
            "Sell": "#f8d7da",   # light red
            "Hold": "#fff3cd",   # light orange/yellow
            "N/A": "#f0f0f0"     # gray fallback
        }
        sentiment_symbols = {
            "Buy": "<span style='color:green;font-size:1.5em;'>⬆️</span>",
            "Sell": "<span style='color:red;font-size:1.5em;'>⬇️</span>",
            "Hold": "<span style='color:orange;font-size:1.5em;'>⎯</span>",
            "N/A": "<span style='color:gray;'>?</span>"
        }
        cols = st.columns(len(asset_classes))
        for i, asset in enumerate(asset_classes):
            with cols[i]:
                sentiment = sentiment_data.get(asset, {}).get('sentiment', 'N/A')
                rationale = sentiment_data.get(asset, {}).get('rationale', 'N/A')
                color = sentiment_colors.get(sentiment, "#f0f0f0")
                symbol = sentiment_symbols.get(sentiment, sentiment_symbols["N/A"])
                text_color = "#222222"
                st.markdown(
                    f"""
                    <div style="background-color:{color};padding:16px;border-radius:10px;text-align:center;">
                        <strong style="color:{text_color};">{asset}</strong><br>
                        <span style="font-size:1.2em;color:{text_color};">{symbol} {sentiment}</span>
                        <div style="font-size:0.9em;margin-top:8px;color:{text_color};">{rationale}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.subheader("Feedback on Sentiment")

    labels = {}
    # Use a form to group the radio buttons and the submit button
    with st.form("feedback_form"):
        cols = st.columns(len(asset_classes))
        for idx, asset in enumerate(asset_classes):
            # Get the default sentiment for the radio button
            asset_data = sentiment_data.get(asset, {})
            default_sentiment = asset_data.get('sentiment', 'Hold')
            default_index = options.index(default_sentiment) if default_sentiment in options else 2

            with cols[idx]:
                st.markdown(f"**{asset}**")
                labels[asset] = st.radio(f"Sentiment for {asset}", options, index=default_index, key=f"label_{asset}", horizontal=True, label_visibility="collapsed")

        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            # Create a simple sentiment dict from the parsed data for comparison
            parsed_sentiments_only = {k: v.get('sentiment', 'Hold') for k, v in sentiment_data.items()}
            # Compare the dictionaries to see if user feedback is different
            is_different = parsed_sentiments_only != labels

            # The user_labels dictionary must be converted to a JSON string
            # to be stored in a single BigQuery column of type STRING.
            errors = store_feedback_bq(PROJECT_ID, DATASET_ID, TABLE_ID, {
                "document": st.session_state.selected_pdf,
                "document_text": st.session_state.document_text,
                "predicted_sentiment": st.session_state.sentiment,  # Raw text from LLM
                "parsed_predicted_sentiment": json.dumps(sentiment_data),  # Parsed original prediction with rationale
                "user_labels": json.dumps(labels),  # User-corrected labels
                "is_different": is_different
            })
            if errors:
                st.error(f"Failed to save feedback: {errors}")
            else:
                st.success("Feedback submitted successfully!")
                # Clear the state to reset the UI for the next analysis
                st.session_state.sentiment = None
                st.session_state.selected_pdf = None
                st.session_state.document_text = None
                st.rerun()

