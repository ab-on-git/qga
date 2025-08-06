source workspace/hackathon/bin/activate

export GOOGLE_APPLICATION_CREDENTIALS="./quantgenadvisors-key.json"
export PROJECT_ID="quantgenadvisors-2"
export LOCATION="us-central1"

gcloud auth activate-service-account project-service-account@quantgenadvisors-2-35de.iam.gserviceaccount.com --key-file=./quantgenadvisors-key.json
cd streamlit-app
streamlit run main.py --server.address=localhost --server.enableCORS=false --server.enableXsrfProtection=false

