#!/bin/bash

# GCP Setup Script for Resume RAG Backend
# This script sets up necessary GCP resources before deployment

set -e

PROJECT_ID="acquired-voice-474914-j2"
REGION="asia-south1"

echo "🔧 Setting up GCP resources..."

# Set the project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "📋 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable storage-component.googleapis.com

# Get project number for service account
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "🔑 Granting permissions to service account: ${SERVICE_ACCOUNT}"

# Grant necessary permissions to the default compute service account
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin" \
  --condition=None

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor" \
  --condition=None

echo "✅ GCP setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Make sure your secrets are in Secret Manager:"
echo "   - GEMINI_API_KEY"
echo "   - PINECONE_API_KEY"
echo "   - MONGODB_URI"
echo "   - GCS_BUCKET"
echo ""
echo "2. Make sure your GCS bucket exists (or create it):"
echo "   gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=${REGION}"
echo ""
echo "3. Run ./deploy.sh to deploy the application"


