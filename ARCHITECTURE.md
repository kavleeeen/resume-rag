# Architecture Documentation

## Overview

The Resume RAG Backend is a microservices-oriented backend system that uses Retrieval-Augmented Generation (RAG) and semantic similarity to match resumes with job descriptions. The architecture follows SOLID principles with clear separation of concerns.

## System Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ HTTP/REST
       │
┌──────▼─────────────────────────────────────────────┐
│              Express Server (server.ts)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Upload  │  │ Comparison│  │   Chat   │          │
│  │ Endpoint │  │ Endpoint  │  │ Endpoint │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
└───────┼──────────────┼─────────────┼────────────────┘
        │              │             │
        │              │             │
┌───────▼─────────────▼─────────────▼────────────────┐
│                  Services Layer                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  DocumentProcessingWorker                    │  │
│  │  - Text Extraction                           │  │
│  │  - Text Cleaning                             │  │
│  │  - Chunking                                  │  │
│  │  - Embedding Generation                      │  │
│  │  - Vector Indexing                           │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  MatchingService                             │  │
│  │  - Semantic Matching                         │  │
│  │  - Skill Matching (with MMR)                │  │
│  │  - Years Matching                            │  │
│  │  - Score Aggregation                         │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │  ChatService                                 │  │
│  │  - RAG Retrieval                             │  │
│  │  - MMR Selection                             │  │
│  │  - LLM Generation                            │  │
│  └──────────────────────────────────────────────┘  │
└───────┬──────────────┬─────────────┬────────────────┘
        │              │             │
        │              │             │
┌───────▼──────────────▼─────────────▼────────────────┐
│              External Services                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ MongoDB  │  │ Pinecone  │  │  Gemini  │         │
│  │ (Data)   │  │ (Vectors) │  │  (LLM)   │         │
│  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Processing Pipeline

**Flow:**
```
Upload → Hash Check → Text Extraction → Cleaning → Chunking → Embedding → Vector Storage
```

**Components:**
- **DocumentReuseService**: Checks file hashes to avoid reprocessing
- **TextExtractionService**: Extracts text from PDF/DOCX/TXT
- **TextCleaningService**: Cleans and normalizes text
- **ChunkingService**: Splits text into semantic chunks (3000 chars, 800 overlap)
- **EmbeddingService**: Generates vector embeddings (768 dimensions)
- **PineconeService**: Stores and queries vectors

**Key Features:**
- Parallel processing for resume and JD
- Hash-based deduplication
- Automatic document vector (docvec) generation
- Metadata preservation for traceability

### 2. Matching Algorithm

**Multi-Factor Scoring:**

1. **Semantic Score (40%)**
   - Uses JD document-level embedding
   - Queries top 20 resume chunks
   - Weighted average with quality/coverage boosts
   - Captures conceptual similarity (e.g., "AWS" vs "EC2 via Kubernetes")

2. **Keyword Score (55%)**
   - Extracts top skills (4-5) and general skills from JD
   - Uses semantic matching with MMR for diversity
   - Applies deduplication to avoid redundant chunks
   - Validates with keyword matching (OR condition)
   - Weighted scoring (top skills × 2, general skills × 1)

3. **Years Score (5%)**
   - Extracts years from resume and JD
   - Calculates ratio-based score
   - Handles edge cases (no years specified)

**Final Score:**
```
finalScore = (0.4 × semanticScore) + (0.55 × keywordScore) + (0.05 × yearsScore)
finalPercent = round(finalScore × 100)
```

### 3. Chat Service (RAG)

**Retrieval-Augmented Generation Flow:**

```
User Question → Embedding → Pinecone Query → MMR Selection → LLM Prompt → Answer
```

**Process:**
1. **Question Embedding**: Convert user question to vector
2. **Retrieval**: Query Pinecone for top 24 chunks (2x for selection)
3. **Deduplication**: Remove duplicate chunks
4. **MMR Selection**: Select top 12 diverse chunks
5. **Prompt Building**: Construct RAG prompt with context
6. **LLM Generation**: Generate answer using Gemini 2.5 Flash

**MMR (Maximal Marginal Relevance):**
- Balances relevance and diversity
- Formula: `MMR = λ × relevance - (1-λ) × max_similarity_to_selected`
- Lambda: 0.6 (60% relevance, 40% diversity)

## Data Models

### Document Record
```typescript
{
  docId: string;
  type: 'resume' | 'job_description';
  filename: string;
  rawText: string;
  status: 'pending' | 'processing' | 'indexed' | 'failed';
  indexedAt?: Date;
  vectorCount?: number;
  fileHash?: string;
}
```

### Comparison Record
```typescript
{
  comparisonId: string;
  resumeDocId: string;
  jdDocId: string;
  status: 'in_progress' | 'completed' | 'failed';
  matchResult?: {
    finalPercent: number;
    semanticScore: number;
    keywordScore: number;
    yearsScore: number;
    matchedSkills: string[];
    missingSkills: string[];
  };
}
```

### Vector Metadata (Pinecone)
```typescript
{
  doc_id: string;           // e.g., "resume:uuid"
  doc_type: 'resume' | 'job_description';
  chunk_index: number;      // -1 for docvec, 0+ for chunks
  text_snippet?: string;    // First 300 chars
  section?: string;
  version: 'v1';
  uploaded_at: string;
}
```

## Service Responsibilities

### DocumentProcessingWorker
- **Single Responsibility**: Process documents end-to-end
- **Dependencies**: TextExtraction, TextCleaning, Chunking, Embedding, Pinecone
- **Key Methods**:
  - `processDocument()`: Main processing pipeline
  - `copyVectorsFromExisting()`: Reuse existing vectors

### MatchingService
- **Single Responsibility**: Calculate resume-JD match scores
- **Dependencies**: PineconeService, EmbeddingService, DatabaseService
- **Key Methods**:
  - `calculateMatch()`: Main matching algorithm
  - `matchSkillsSemantically()`: Skill matching with MMR
  - `computeSemanticScoreWithBoosts()`: Semantic score calculation

### ChatService
- **Single Responsibility**: Provide RAG-powered chat
- **Dependencies**: EmbeddingService, PineconeService, DatabaseService
- **Key Methods**:
  - `chat()`: Main chat interface
  - `retrieveChunks()`: Semantic retrieval
  - `mmrSelect()`: Diversity selection
  - `buildRAGPrompt()`: Context construction

### DatabaseService
- **Single Responsibility**: Database operations
- **Collections**:
  - `documents`: Document records
  - `jds`: Job description records with skills
  - `comparisons`: Comparison results
  - `chat_sessions`: Chat conversation history
  - `file_hashes`: Hash-based deduplication

## Design Patterns

### 1. Service Layer Pattern
Each service encapsulates a specific domain:
- Clear boundaries
- Testable in isolation
- Reusable components

### 2. Repository Pattern
DatabaseService acts as a repository:
- Abstracts database operations
- Provides consistent interface
- Handles connection management

### 3. Strategy Pattern
Matching algorithm uses different strategies:
- Semantic matching
- Keyword matching
- Years matching
- Configurable weights

### 4. Factory Pattern
Service initialization:
- Conditional service creation
- Graceful degradation
- Error handling

## Data Flow

### Upload & Processing Flow

```
1. Client uploads resume + JD
   ↓
2. Server generates IDs and creates comparison record
   ↓
3. Files uploaded to GCS (optional)
   ↓
4. Background processing starts:
   a. Calculate file hashes
   b. Check for existing documents
   c. Process documents (if needed):
      - Extract text
      - Clean text
      - Chunk text
      - Generate embeddings
      - Store in Pinecone
   ↓
5. Wait for indexing completion
   ↓
6. Calculate match scores
   ↓
7. Update comparison with results
```

### Chat Flow

```
1. User asks question
   ↓
2. Generate question embedding
   ↓
3. Query Pinecone for relevant chunks
   ↓
4. Deduplicate chunks
   ↓
5. Apply MMR for diversity
   ↓
6. Build RAG prompt with:
   - Selected chunks
   - Conversation history
   - User question
   ↓
7. Generate answer with LLM
   ↓
8. Store messages in database
   ↓
9. Return answer to user
```

## Performance Optimizations

### 1. Parallel Processing
- Resume and JD processed in parallel
- Embeddings generated concurrently (30 parallel requests)
- GCS uploads in parallel

### 2. Caching & Reuse
- Document hash-based deduplication
- Skill embeddings cached in database
- Vector reuse for identical files

### 3. Efficient Retrieval
- MMR for diverse chunk selection
- Deduplication to avoid redundant chunks
- Batch operations for Pinecone

### 4. Database Optimization
- Connection pooling (max 20, min 5)
- Indexed file hashes for fast lookups
- Efficient query patterns

## Error Handling

### Graceful Degradation
- Services initialize conditionally
- Missing services don't crash server
- Health endpoint always available

### Retry Logic
- Document processing: 3 retries with exponential backoff
- Embedding generation: Network error retries
- Database connection: Automatic reconnection

### Error States
- Document status: `pending` → `processing` → `indexed` / `failed`
- Comparison status: `in_progress` → `completed` / `failed`
- Clear error messages in responses

## Security Considerations

### Input Validation
- File type validation (PDF, DOCX, TXT only)
- File size limits (10MB)
- Question validation for chat

### Data Privacy
- Files stored securely in GCS
- Database connections encrypted
- API keys in environment variables

### CORS
- Configured for specific origins
- Supports localhost and custom domains

## Scalability

### Horizontal Scaling
- Stateless services (except database)
- Cloud Run auto-scaling
- Pinecone handles vector queries at scale

### Vertical Scaling
- Configurable embedding concurrency
- Adjustable batch sizes
- Connection pool tuning

## Monitoring & Observability

### Logging
- High-level operation logs
- Error logging with context
- Performance metrics (match scores, processing times)

### Health Checks
- `/health` endpoint for monitoring
- Database connection status
- Service initialization status

## Future Enhancements

### Potential Improvements
1. **Caching Layer**: Redis for frequently accessed data
2. **Queue System**: Background job queue for processing
3. **Analytics**: Track match accuracy and user feedback
4. **Multi-language Support**: Handle resumes in different languages
5. **Advanced Matching**: Machine learning-based scoring
6. **Real-time Updates**: WebSocket support for live updates

## Dependencies

### External Services
- **MongoDB**: Document storage
- **Pinecone**: Vector database
- **Google Gemini**: LLM and embeddings
- **Google Cloud Storage**: File storage (optional)

### Key Libraries
- **Express**: Web framework
- **Multer**: File upload handling
- **pdf-parse**: PDF text extraction
- **mammoth**: DOCX text extraction
- **@pinecone-database/pinecone**: Vector database client
- **@google/generative-ai**: Gemini API client

## Code Quality

### SOLID Principles
- **Single Responsibility**: Each service has one clear purpose
- **Open/Closed**: Extensible through configuration
- **Liskov Substitution**: Consistent interfaces
- **Interface Segregation**: Focused service interfaces
- **Dependency Inversion**: Services depend on abstractions

### Best Practices
- TypeScript for type safety
- Error handling at all levels
- Comprehensive logging
- Test coverage for critical paths
- Clean code with meaningful names

