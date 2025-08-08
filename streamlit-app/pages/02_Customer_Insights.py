import streamlit as st
import pandas as pd
import re
import json
from google.cloud import bigquery
import os
from datetime import datetime, timedelta

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = "portfolios" # Hardcode correct table to avoid env conflicts

# Import utility functions from the utils.py file
from utils import (
    load_vector_store,
    load_embedding_model,
    generate_gemini_content,
    find_relevant_insights_batch
)

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("CIO Portfolio Advisor")

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_portfolio_data(project_id=PROJECT_ID, dataset_id=DATASET_ID, table_id=TABLE_ID):
    """
    Loads customer portfolio data from a BigQuery table and returns it as a DataFrame.
    The table name is hardcoded to 'customer_portfolio' to avoid conflicts with other table IDs.
    """
    try:
        client = bigquery.Client(project=project_id)
        query = f"""
            SELECT
                CustomerName,
                InvestmentObjective,
                RiskLevel,
                CurrentAllocation,
                TargetAllocation
            FROM `{project_id}.{dataset_id}.{table_id}` -- Use the correct table name
            ORDER BY CustomerName
        """
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Failed to load customer data from BigQuery: {e}")
        return pd.DataFrame()

# Use st.cache_resource for loading models and resources that should persist across sessions.
@st.cache_resource
def load_models_and_store():
    """Loads and caches the embedding model and vector store."""
    embedding_model = load_embedding_model()
    index, metadata = load_vector_store()
    return embedding_model, index, metadata

portfolio_df = load_portfolio_data()
embedding_model, index, metadata = load_models_and_store()

# --- Main App Logic ---
# Stop the app if essential data or models failed to load.
if portfolio_df.empty or index is None:
    st.warning("Application cannot start due to missing data or vector store. Please check the error messages above.")
    st.stop()

# --- UI Section ---
st.header("1. Select a Client")
clients = portfolio_df["CustomerName"].unique()
selected_client = st.selectbox(
    "Select a client to view their portfolio and generate insights:",
    clients,
    label_visibility="collapsed"
)

# Get the selected client's profile
client_profile_df = portfolio_df[portfolio_df["CustomerName"] == selected_client].copy()

st.header(f"2. Portfolio Overview for {selected_client}")
if not client_profile_df.empty:
    st.dataframe(
        client_profile_df[[
            "InvestmentObjective", "RiskLevel", "CurrentAllocation", "TargetAllocation"
        ]],
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning(f"No portfolio data found for {selected_client}.")

st.header("3. Generate CIO-Powered Recommendations")

if st.button("Analyze Holdings with CIO Insight", key="analyze_holdings"):
    if client_profile_df.empty:
        st.warning(f"No holdings found for {selected_client}.")
    else:
        with st.spinner("Analyzing client profile against recent CIO insights..."):
            # --- Get client's profile ---
            client_profile = client_profile_df.iloc[0]
            investment_objective = client_profile['InvestmentObjective']
            risk_level = client_profile['RiskLevel']
            current_allocation = client_profile['CurrentAllocation']
            target_allocation = client_profile['TargetAllocation']

            # --- ASSET CLASS LEVEL OPTIMIZATION ---
            # 1. Define asset classes to analyze.
            unique_asset_classes = ["Equity", "Fixed Income", "Commodities", "Real Estate", "Crypto"]

            # 2. Create a list of queries for each unique asset class.
            asset_class_queries = [
                f"What is the CIO outlook for the {asset_class} asset class?"
                for asset_class in unique_asset_classes
            ]

            # 3. Retrieve all relevant contexts in a single batch call.
            # FIX: Pass the list of queries, not a single string.
            # Use date filtering to get the most recent insights.
            all_context_data = find_relevant_insights_batch(
                asset_class_queries, index, metadata, embedding_model, days_filter=180
            )

            # 4. Build a single prompt for all asset classes to generate recommendations in one call.
            all_asset_class_info = []
            for i, asset_class in enumerate(unique_asset_classes):
                context = all_context_data[i]["context"]
                all_asset_class_info.append(
                    f"{context}"
                )
            
            #asset_class_info_string = "\n\n---\n\n".join(all_asset_class_info)

            final_prompt = f"""
            You are a CIO-level investment advisor. Your task is to provide a "Buy", "Sell", or "Hold" recommendation for a client's portfolio based on their profile and the latest CIO insights.

            **Client Profile:**
            - Investment Objective: {investment_objective}
            - Risk Level: {risk_level}
            - Current Allocation: {current_allocation}
            - Target Allocation: {target_allocation}

            **Context to use for response:**
            ---
            {all_context_data}
            ---

            **Your Task:**
            Based on the client's profile and the provided context, generate a recommendation for each of the following asset classes: {', '.join(unique_asset_classes)}.
            Your response must be a single, valid JSON object. The keys of the object should be the asset class names. Each value should be an object with two keys: "action" and "rationale".
            - "action" must be one of "Buy", "Sell", or "Hold".
            - "rationale" must be a concise, one-sentence explanation that considers both the CIO insight and the client's specific profile.

            Example Response Format:
            ```json
            {{
              "Equity": {{
                "action": "Buy",
                "rationale": "Positive market outlook and strong earnings growth."
              }},
              "Fixed Income": {{
                "action": "Hold",
                "rationale": "Yields are stable but potential for volatility remains."
              }}
            }}
            ```
            """

            st.session_state.recommendations = {}
            try:
                # 5. Make a single API call for all asset classes
                response_text = generate_gemini_content(final_prompt)
                
                # 6. Parse the JSON response, handling potential markdown code blocks
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
                json_str = json_match.group(1) if json_match else response_text
                parsed_response = json.loads(json_str)

                # 7. Populate the recommendations dictionary from the single response
                st.session_state.recommendations = parsed_response
                st.session_state.context_data = {ac: all_context_data[i] for i, ac in enumerate(unique_asset_classes)}
                st.rerun()

            except json.JSONDecodeError:
                st.error("Failed to parse the JSON response from the model. The response was not valid JSON.")
                st.text_area("Model Response:", response_text, height=200)
            except Exception as e:
                st.error(f"An error occurred during insight generation: {e}")

# Display results if they exist in the session state
if 'recommendations' in st.session_state and st.session_state.recommendations:
    st.subheader("Buy/Sell/Hold Recommendations by Asset Class")

    summary_list = []
    for asset_class, rec_data in st.session_state.recommendations.items():
        if isinstance(rec_data, dict):
            summary_list.append({
                "Asset Class": asset_class,
                "Recommendation": rec_data.get("action", "N/A"),
                "Rationale": rec_data.get("rationale", "N/A")
            })

    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Show Video Here
    # Construct an absolute path to the video file relative to the script's location.
    # This makes the file lookup independent of the current working directory.
    st.subheader("Video briefing ...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join("..", script_dir, "5356023141593125734-sample_0.mp4")
    if os.path.exists(video_path):
        with st.spinner("Loading video..."):
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
    else:
        st.warning(f"Sample video file not found at the expected path: `{video_path}`. Please ensure `sample_0.mp4` is in the same directory as the script.")

    with st.expander("View Detailed Context and Client Allocations"):
        for asset_class, rec_data in st.session_state.recommendations.items():
            if not isinstance(rec_data, dict): continue

            st.markdown(f"#### {asset_class}")
            st.markdown(f"**Recommendation:** {rec_data.get('action', 'N/A')}")
            st.markdown(f"**Rationale:** {rec_data.get('rationale', 'No rationale provided.')}")

            # Display client's allocation for this asset class
            #st.markdown("**Client's Allocation:**")
            try:
                current_alloc = json.loads(client_profile_df.iloc[0]['CurrentAllocation'])
                target_alloc = json.loads(client_profile_df.iloc[0]['TargetAllocation'])
                #st.metric(label=f"Current {asset_class} Allocation", value=f"{current_alloc.get(asset_class, 0)}%")
                #st.metric(label=f"Target {asset_class} Allocation", value=f"{target_alloc.get(asset_class, 0)}%")
            except (json.JSONDecodeError, KeyError, IndexError):
                st.caption("Could not display allocation details.")

            # Display context from RAG
            context_data = st.session_state.context_data.get(asset_class, {})
            st.markdown("**Source Context Used for Recommendation:**")
            st.info(context_data.get('context', 'No context found.'))
            st.markdown("**Source Document(s):**")
            st.caption(", ".join(context_data.get("sources", ["N/A"])))
            st.markdown("---")