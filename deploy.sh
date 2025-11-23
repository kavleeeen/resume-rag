#!/usr/bin/env bash
set -euo pipefail

# Minimal Cloud Run deploy helper
# Requirements: gcloud CLI installed and authenticated

PROJECT_ID=${PROJECT_ID:-"$(gcloud config get-value project 2>/dev/null || true)"}
REGION=${REGION:-asia-south1}
SERVICE_NAME=${SERVICE_NAME:-resume-rag-be}
# Secret Manager secret names
MONGODB_URI_SECRET_NAME=${MONGODB_URI_SECRET_NAME:-MONGODB_URI}
GCS_BUCKET_SECRET_NAME=${GCS_BUCKET_SECRET_NAME:-GCS_BUCKET}
IMAGE=gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID is not set and not configured in gcloud."
  echo "Set it via: export PROJECT_ID=your-project-id"
  exit 1
fi

echo "Building image: ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}"

echo "Deploying to Cloud Run: ${SERVICE_NAME} (${REGION})"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 4 \
  --memory 4Gi \
  --timeout 3600 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars "EMBEDDING_DIMENSION=768,EMBEDDING_MODEL=text-embedding-004,PINECONE_INDEX_NAME=resume-rag-index,EMBEDDING_BATCH_SIZE=100,EMBEDDING_MAX_CONCURRENT=30,EMBEDDING_REQUEST_DELAY_MS=0" \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest,MONGODB_URI=MONGODB_URI:latest,GCS_BUCKET=GCS_BUCKET:latest" \

echo "✅ Deployment complete!"
echo "🌐 Service URL:"
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format="value(status.url)"

