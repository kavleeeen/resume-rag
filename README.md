# Resume RAG Backend

A sophisticated backend service for intelligent resume-to-job-description matching using Retrieval-Augmented Generation (RAG) and semantic similarity. The system uses vector embeddings, semantic search, and LLM-powered chat to provide accurate resume matching and interactive Q&A capabilities.

## 🚀 Features

- **Intelligent Resume Matching**: Multi-factor matching algorithm combining:
  - Semantic similarity (40% weight)
  - Keyword/skill matching (55% weight)
  - Years of experience matching (5% weight)
- **RAG-Powered Chat**: Ask questions about resumes and get contextual answers
- **Document Processing**: Automatic extraction, chunking, and embedding generation
- **Smart Document Reuse**: Hash-based deduplication to avoid reprocessing identical documents
- **Skill Extraction**: LLM-powered extraction of critical and general skills from job descriptions
- **Vector Search**: Pinecone-based semantic search with MMR (Maximal Marginal Relevance) for diversity

## 📋 Prerequisites

- Node.js 18+ and npm
- MongoDB (Atlas recommended)
- Pinecone account and index
- Google Gemini API key
- Google Cloud Storage (optional, for file storage)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd resume-rag-be
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Server
   PORT=5003

   # MongoDB
   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/resume?retryWrites=true&w=majority

   # Pinecone
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=resume-rag-index
   EMBEDDING_DIMENSION=768

   # Google Generative AI (Gemini)
   GEMINI_API_KEY=your-gemini-api-key
   EMBEDDING_MODEL=text-embedding-004

   # Google Cloud Storage (optional)
   GOOGLE_CLOUD_PROJECT_ID=your-project-id
   GCS_BUCKET=your-bucket-name
   GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

   # Optional: Embedding concurrency
   EMBEDDING_MAX_CONCURRENT=30
   EMBEDDING_REQUEST_DELAY_MS=0
   ```

4. **Build the project**
   ```bash
   npm run build
   ```

5. **Start the server**
   ```bash
   npm start
   ```

   For development with hot reload:
   ```bash
   npm run dev
   ```

## 📡 API Endpoints

### Health Check
```http
GET /health
```
Returns server health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### Upload Documents
```http
POST /api/upload
Content-Type: multipart/form-data
```

Upload a resume and job description for comparison.

**Request:**
- `resume`: File (PDF, DOCX, or TXT)
- `jobDescription`: File (PDF, DOCX, or TXT)

**Response:**
```json
{
  "message": "Files uploaded successfully. Processing started.",
  "comparisonId": "comparison:uuid",
  "status": "in_progress"
}
```

**Status Codes:**
- `202 Accepted`: Upload successful, processing started
- `400 Bad Request`: Missing or invalid files

### Get Comparison Results
```http
GET /api/comparison/:id
```

Retrieve comparison results and match scores.

**Response:**
```json
{
  "success": true,
  "comparison": {
    "comparisonId": "comparison:uuid",
    "resumeDocId": "resume:uuid",
    "jdDocId": "jd:uuid",
    "status": "completed",
    "matchResult": {
      "finalPercent": 85,
      "semanticScore": 0.82,
      "keywordScore": 0.88,
      "yearsScore": 1.0,
      "resumeYears": 5,
      "jdRequiredYears": 3,
      "matchedSkills": ["React", "Node.js", "AWS"],
      "missingSkills": ["Docker"]
    }
  }
}
```

**Status Codes:**
- `200 OK`: Comparison found
- `404 Not Found`: Comparison not found

### Chat with Resume
```http
POST /api/comparison/:id/chat
Content-Type: application/json
```

Ask questions about the resume using RAG.

**Request:**
```json
{
  "question": "What programming languages does the candidate know?"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "The candidate has experience with JavaScript, TypeScript, Python, and Java..."
}
```

**Status Codes:**
- `200 OK`: Chat response generated
- `400 Bad Request`: Missing or invalid question
- `503 Service Unavailable`: Chat service not initialized

## 🧪 Testing

Run tests:
```bash
npm test
```

Run tests in watch mode:
```bash
npm run test:watch
```

Run tests with coverage:
```bash
npm run test:coverage
```

## 🏗️ Project Structure

```
resume-rag-be/
├── services/              # Core business logic
│   ├── ChatService.ts           # RAG-powered chat
│   ├── MatchingService.ts        # Resume-JD matching algorithm
│   ├── DocumentProcessingWorker.ts # Document processing pipeline
│   ├── EmbeddingService.ts       # Vector embedding generation
│   ├── PineconeService.ts        # Vector database operations
│   ├── DatabaseService.ts        # MongoDB operations
│   └── ...
├── utils/                 # Shared utilities
│   └── scoreNormalization.ts    # Score normalization logic
├── types/                 # TypeScript type definitions
├── __tests__/             # Test files
├── server.ts              # Express server and API routes
└── package.json
```

## 🔧 Configuration

### Embedding Settings
- **Model**: `text-embedding-004` (Google Gemini)
- **Dimension**: 768 (configurable via `EMBEDDING_DIMENSION`)
- **Concurrency**: 30 parallel requests (configurable via `EMBEDDING_MAX_CONCURRENT`)

### Matching Algorithm Weights
- **Semantic Score**: 40% (conceptual similarity)
- **Keyword Score**: 55% (skill matching)
- **Years Score**: 5% (experience matching)

### Chat Settings
- **Model**: `gemini-2.5-flash`
- **Temperature**: 0.2 (for consistent responses)
- **Max History**: 10 messages
- **Top K**: 12 chunks (with MMR for diversity)

## 🚢 Deployment

See [DEPLOY.md](./DEPLOY.md) for detailed deployment instructions to Google Cloud Run.

Quick deployment:
```bash
./deploy.sh
```

## 📊 How It Works

1. **Document Upload**: Resume and JD files are uploaded and stored
2. **Text Extraction**: PDF/DOCX files are parsed to extract text
3. **Chunking**: Text is split into semantic chunks (3000 chars, 800 overlap)
4. **Embedding**: Chunks are converted to vector embeddings
5. **Indexing**: Vectors are stored in Pinecone for semantic search
6. **Matching**: Multi-factor algorithm calculates match percentage
7. **Chat**: RAG retrieves relevant chunks and generates answers

## 🔍 Key Technologies

- **Express.js**: REST API framework
- **MongoDB**: Document database
- **Pinecone**: Vector database for semantic search
- **Google Gemini**: LLM for embeddings and chat
- **TypeScript**: Type-safe development

## 📝 License

ISC

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions, please open an issue on GitHub.

