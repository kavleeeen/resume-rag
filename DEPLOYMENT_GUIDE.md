# GCP Deployment Guide for Resume RAG Backend

## Overview
This guide outlines the steps and information needed to deploy the Resume RAG Backend to Google Cloud Platform.

## Recommended Deployment Option: **Cloud Run**
- **Why Cloud Run?**
  - Serverless and auto-scaling
  - Pay only for what you use
  - Easy container deployment
  - Built-in HTTPS
  - Handles background processing well
  - Supports long-running requests (up to 60 minutes)

## Prerequisites & Information Needed

### 1. GCP Project Setup
- [ ] **GCP Project ID**: `acquired-voice-474914-j2` (already in code) or new project
- [ ] **Billing Account**: Ensure billing is enabled
- [ ] **APIs to Enable**:
  - Cloud Run API
  - Cloud Storage API
  - Cloud Build API
  - Secret Manager API (recommended for secrets)

### 2. Environment Variables & Secrets

#### Required Environment Variables:
```bash
# Server
PORT=8080  # Cloud Run uses 8080 by default

# Google Cloud
GOOGLE_CLOUD_PROJECT_ID=<your-project-id>
GCS_BUCKET=<your-bucket-name>
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json  # Only if using file-based auth

# MongoDB
MONGODB_URI=<your-mongodb-connection-string>

# Pinecone
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_INDEX_NAME=resume-rag-index  # or your index name
EMBEDDING_DIMENSION=768

# Google Generative AI (Gemini)
GEMINI_API_KEY=<your-gemini-api-key>

# Optional
EMBEDDING_MODEL=text-embedding-004  # default
```

#### Questions to Answer:
1. **MongoDB**: 
   - Are you using MongoDB Atlas? (Recommended for GCP)
   - Or self-hosted MongoDB?
   - Connection string format: `mongodb+srv://user:pass@cluster.mongodb.net/resume?retryWrites=true&w=majority`

2. **GCS Bucket**:
   - Bucket name: `resume-rag-files` (or your preferred name)
   - Region: Should match Cloud Run region (e.g., `us-central1`)
   - Storage class: `STANDARD` (default) or `NEARLINE` for cost savings

3. **Service Account**:
   - Do you want to use Workload Identity (recommended) or service account JSON file?
   - Required permissions:
     - `Storage Object Admin` (for GCS uploads)
     - `Secret Manager Secret Accessor` (if using Secret Manager)

### 3. Service Account Setup

**Option A: Workload Identity (Recommended)**
- Cloud Run automatically uses the service account attached to the service
- No need for `GOOGLE_APPLICATION_CREDENTIALS` file
- More secure and follows GCP best practices

**Option B: Service Account JSON File**
- Create service account in GCP Console
- Download JSON key
- Store in Secret Manager or as environment variable (less secure)

### 4. Docker Configuration

#### Files Needed:
- [ ] **Dockerfile** (to be created)
- [ ] **.dockerignore** (to be created)
- [ ] **cloudbuild.yaml** (optional, for CI/CD)

#### Dockerfile Requirements:
- Base image: `node:20-alpine` or `node:20-slim`
- Build step: `npm install` and `npm run build`
- Runtime: `node dist/server.js`
- Port: `8080`
- Health check endpoint (if needed)

### 5. Cloud Run Configuration

#### Service Settings:
- **Service Name**: `resume-rag-backend`
- **Region**: `us-central1` (or your preferred region)
- **CPU**: 1-2 vCPU (depending on load)
- **Memory**: 2-4 GB (LLM calls can be memory-intensive)
- **Min Instances**: 0 (for cost savings) or 1 (for faster cold starts)
- **Max Instances**: 10-100 (based on expected load)
- **Timeout**: 60 minutes (for background processing)
- **Concurrency**: 80 (default, adjust based on needs)
- **Request Timeout**: 300 seconds (5 minutes) for file uploads

#### Environment Variables:
- Set all required env vars in Cloud Run console or via gcloud CLI
- Use Secret Manager for sensitive values (API keys, MongoDB URI)

### 6. Security Considerations

#### Secrets Management:
- [ ] Store sensitive values in **Secret Manager**:
  - `GEMINI_API_KEY`
  - `PINECONE_API_KEY`
  - `MONGODB_URI`
  - Service account JSON (if using file-based auth)

#### IAM Roles:
- [ ] Service account needs:
  - `roles/storage.objectAdmin` (for GCS)
  - `roles/secretmanager.secretAccessor` (for secrets)
  - `roles/run.invoker` (for Cloud Run)

#### CORS:
- [ ] Configure CORS in `server.ts` to allow your frontend domain
- [ ] Update `allowedOrigins` in CORS middleware

### 7. Networking

#### Questions:
1. **Frontend Domain**: What domain will call this API?
   - Update CORS settings accordingly
   - Example: `https://your-frontend.com`

2. **Internal vs External**:
   - Will this be called from external frontend? → Public Cloud Run service
   - Only internal? → Private Cloud Run service with VPC connector

3. **Custom Domain** (Optional):
   - Do you want a custom domain?
   - Requires Cloud Load Balancer setup

### 8. Monitoring & Logging

#### Setup:
- [ ] **Cloud Logging**: Automatic (enabled by default)
- [ ] **Cloud Monitoring**: Set up alerts for:
  - Error rate
  - Request latency
  - Memory usage
  - CPU usage
- [ ] **Error Reporting**: Enable for production errors

### 9. Cost Estimation

#### Monthly Costs (Approximate):
- **Cloud Run**: 
  - $0.40 per million requests
  - $0.00002400 per vCPU-second
  - $0.00000250 per GiB-second
  - Example: 100K requests/month, 1 vCPU, 2GB RAM, 5s avg = ~$10-20/month

- **Cloud Storage**:
  - $0.020 per GB/month (Standard)
  - $0.005 per GB/month (Nearline)
  - Example: 10GB = $0.20/month (Standard)

- **Cloud Build**: 
  - First 120 build-minutes/day free
  - $0.003 per build-minute after

- **Secret Manager**:
  - $0.06 per secret version per month
  - $0.03 per 10,000 operations

### 10. Deployment Steps (High-Level)

1. **Prepare Code**:
   - Create Dockerfile
   - Create .dockerignore
   - Update CORS settings
   - Test build locally

2. **Set Up GCP Resources**:
   - Create GCS bucket
   - Create service account
   - Set up Secret Manager secrets
   - Enable required APIs

3. **Build & Deploy**:
   - Build Docker image (locally or via Cloud Build)
   - Push to Container Registry or Artifact Registry
   - Deploy to Cloud Run

4. **Configure**:
   - Set environment variables
   - Attach service account
   - Configure scaling
   - Set up monitoring

5. **Test**:
   - Test API endpoints
   - Verify file uploads to GCS
   - Test background processing
   - Verify database connections

## Files to Create

### 1. Dockerfile
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY --from=builder /app/dist ./dist
EXPOSE 8080
CMD ["node", "dist/server.js"]
```

### 2. .dockerignore
```
node_modules
dist
.env
.env.local
*.log
.git
.gitignore
__tests__
*.test.ts
*.test.js
jest.config.js
tsconfig.json
README.md
DEPLOYMENT_GUIDE.md
```

### 3. .gcloudignore (for gcloud CLI)
```
node_modules
dist
.env
.git
__tests__
*.test.ts
*.test.js
```

## Questions to Answer Before Deployment

1. **What is your GCP Project ID?**
   - Current default: `acquired-voice-474914-j2`
   - Confirm or provide new one

2. **What is your MongoDB connection string?**
   - Format: `mongodb+srv://...` or `mongodb://...`

3. **What is your GCS bucket name?**
   - Will create if doesn't exist
   - Region preference?

4. **What is your frontend domain?**
   - For CORS configuration

5. **Do you have all API keys?**
   - Gemini API key
   - Pinecone API key

6. **Preferred Cloud Run region?**
   - Options: `us-central1`, `us-east1`, `europe-west1`, etc.

7. **Expected traffic?**
   - Requests per day/month
   - Peak concurrent users
   - File upload sizes

8. **Budget constraints?**
   - Min instances (0 = cost savings, 1 = faster response)
   - Max instances

9. **Do you want CI/CD?**
   - GitHub Actions?
   - Cloud Build triggers?
   - Manual deployment?

10. **Monitoring preferences?**
    - Alert thresholds
    - Notification channels (email, Slack, etc.)

## Next Steps

Once you provide the answers to the questions above, I can:
1. Create the Dockerfile and related files
2. Generate deployment scripts
3. Create Cloud Run configuration
4. Set up Secret Manager integration
5. Create CI/CD pipeline (if needed)

