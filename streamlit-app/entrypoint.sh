#!/bin/sh
# entrypoint.sh

# Cloud Run provides the PORT environment variable.
# Use 8080 as a fallback for local testing if PORT is not set.
# Note: The error mentions 8501, so Cloud Run is indeed setting it.
# Your app *must* use this specific variable.
LISTEN_PORT=${PORT:-8080}

# Run Streamlit.
# --server.port: Tells Streamlit to listen on the given port.
# --server.address 0.0.0.0: Tells Streamlit to listen on all available network interfaces.
#                           This is CRITICAL for Docker containers.
# --server.enableCORS false: Recommended for Cloud Run to avoid CORS issues.
# --server.enableXsrfProtection false: Recommended for publicly accessible apps.
streamlit run main.py \
    --server.port "$LISTEN_PORT" \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false
