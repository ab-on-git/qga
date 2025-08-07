export GOOGLE_APPLICATION_CREDENTIALS="/Users/as/ab-on-git/qga_repo/streamlit-app/key.json"
export PROJECT_ID="quantgenadvisors-2-35de"
export LOCATION="us-central1"

#gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
#cd streamlit-app
#streamlit run main.py --server.address=localhost --server.enableCORS=false --server.enableXsrfProtection=false

# --- Configuration for your application ---
# Ensure this Project ID matches the project where your Vertex AI API is enabled
# and where your service account has the "Vertex AI User" role.
export BUCKET_NAME="hackathon-qga"
export DATASET_ID="qga"
export TABLE_ID="feedback"