# Use an explicit amd64 Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


ENV BUCKET_NAME="hackathon-qga"
ENV DATASET_ID="qga"
ENV TABLE_ID="feedback"
ENV PROJECT_ID="quantgenadvisors-2-35de"
ENV LOCATION="us-central1"
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/key.json"

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Make entrypoint.sh executable
RUN chmod +x entrypoint.sh

# Expose Streamlit port
EXPOSE 8080

# Set Streamlit to wide layout by default (optional)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Use entrypoint.sh to start the app
ENTRYPOINT ["./entrypoint.sh"]