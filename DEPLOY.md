# Deployment Instructions

## Prerequisites

1. **Install gcloud CLI**: https://cloud.google.com/sdk/docs/install
2. **Install Docker**: https://docs.docker.com/get-docker/
3. **Authenticate with GCP**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

## Step 1: Authenticate with GCP

```bash
gcloud auth login
gcloud auth application-default login
```

## Step 2: Set up GCP Resources

Run the setup script to enable APIs and configure permissions:

```bash
./setup-gcp.sh
```

This will:
- Enable required GCP APIs
- Grant necessary permissions to the service account
- Set up IAM roles

## Step 3: Verify Secrets in Secret Manager

Make sure these secrets exist in Secret Manager:
- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `MONGODB_URI`
- `GCS_BUCKET`

To check:
```bash
gcloud secrets list --project=acquired-voice-474914-j2
```

## Step 4: Create GCS Bucket (if not exists)

```bash
# Get bucket name from Secret Manager first
BUCKET_NAME=$(gcloud secrets versions access latest --secret=GCS_BUCKET --project=acquired-voice-474914-j2)

# Create bucket if it doesn't exist
gsutil ls -b gs://${BUCKET_NAME} || gsutil mb -l asia-south1 gs://${BUCKET_NAME}
```

## Step 5: Deploy to Cloud Run

```bash
./deploy.sh
```

This will:
- Build the Docker image
- Push to Container Registry
- Deploy to Cloud Run in `asia-south1` region
- Configure secrets and environment variables

## Step 6: Verify Deployment

After deployment, you'll get a service URL. Test it:

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe resume-rag-backend --region=asia-south1 --format="value(status.url)")

# Test health endpoint
curl ${SERVICE_URL}/health
```

## Manual Deployment (Alternative)

If you prefer to deploy manually:

```bash
# Set variables
PROJECT_ID="acquired-voice-474914-j2"
REGION="asia-south1"
SERVICE_NAME="resume-rag-backend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Build and push
docker build -t ${IMAGE_NAME}:latest .
docker push ${IMAGE_NAME}:latest

# Deploy
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --min-instances 0 \
  --set-env-vars "PORT=8080,EMBEDDING_DIMENSION=768,EMBEDDING_MODEL=text-embedding-004,PINECONE_INDEX_NAME=resume-rag-index" \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest,MONGODB_URI=MONGODB_URI:latest,GCS_BUCKET=GCS_BUCKET:latest"
```

## Using Cloud Build (CI/CD)

You can also use Cloud Build for automated deployments:

```bash
gcloud builds submit --config cloudbuild.yaml
```

## Configuration

### Environment Variables
- `PORT`: 8080 (Cloud Run default)
- `EMBEDDING_DIMENSION`: 768
- `EMBEDDING_MODEL`: text-embedding-004
- `PINECONE_INDEX_NAME`: resume-rag-index

### Secrets (from Secret Manager)
- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `MONGODB_URI`
- `GCS_BUCKET`

### CORS Configuration
The service allows requests from:
- `http://localhost:*` (any port)
- `https://*.kavleen.in`
- `https://kavleen.in`

## Troubleshooting

### Check logs
```bash
gcloud run services logs read resume-rag-backend --region=asia-south1
```

### Check service status
```bash
gcloud run services describe resume-rag-backend --region=asia-south1
```

### Update service
```bash
./deploy.sh
```

### Delete service
```bash
gcloud run services delete resume-rag-backend --region=asia-south1
```


