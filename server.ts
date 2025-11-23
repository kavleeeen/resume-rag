import express, { Request, Response } from 'express';
import multer, { FileFilterCallback } from 'multer';
import cors from 'cors';
import dotenv from 'dotenv';
import { db } from './services/DatabaseService';
import { DocumentProcessingWorker } from './services/DocumentProcessingWorker';
import { MatchingService } from './services/MatchingService';
import { EmbeddingService } from './services/EmbeddingService';
import { PineconeService } from './services/PineconeService';
import { GCStorageService } from './services/GCStorageService';
import { ChatService } from './services/ChatService';
import { DocumentReuseService } from './services/DocumentReuseService';
import { v4 as uuidv4 } from 'uuid';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5003;

// Initialize services with error handling
let geminiApiKey: string | undefined;
let pineconeApiKey: string | undefined;
let embeddingService: EmbeddingService | null = null;
let pineconeService: PineconeService | null = null;
let matchingService: MatchingService | null = null;
let worker: DocumentProcessingWorker | null = null;
let gcsService: GCStorageService | null = null;
let chatService: ChatService | null = null;

try {
  geminiApiKey = process.env.GEMINI_API_KEY;
  pineconeApiKey = process.env.PINECONE_API_KEY;
  const pineconeIndexName = process.env.PINECONE_INDEX_NAME || 'resume-rag-index';
  const embeddingDimension = parseInt(process.env.EMBEDDING_DIMENSION || '768', 10);

  if (!geminiApiKey) {
    console.error('[Server] WARNING: GEMINI_API_KEY environment variable is not set');
  }
  if (!pineconeApiKey) {
    console.error('[Server] WARNING: PINECONE_API_KEY environment variable is not set');
  }

  // Connect to database asynchronously - don't block startup
  db.connect().catch(err => {
    console.error('[Server] Database connection failed:', err);
    // Don't exit - allow the server to start and retry connections on first request
  });

  if (geminiApiKey && pineconeApiKey) {
    embeddingService = new EmbeddingService(geminiApiKey, embeddingDimension);
    pineconeService = new PineconeService(pineconeApiKey, pineconeIndexName, embeddingDimension);
    matchingService = new MatchingService(pineconeService, embeddingService);
    worker = new DocumentProcessingWorker(
      geminiApiKey,
      pineconeApiKey,
      pineconeIndexName,
      embeddingDimension
    );
    chatService = new ChatService(geminiApiKey, embeddingService, pineconeService);
  }

  try {
    gcsService = new GCStorageService();
  } catch (error) {
    console.error('[Server] GCS service initialization failed:', error);
  }
} catch (error) {
  console.error('[Server] Service initialization error:', error);
  // Continue - health endpoint should still work
}

const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024
  },
  fileFilter: (_req: Request, file: Express.Multer.File, cb: FileFilterCallback) => {
    const allowedMimes = [
      'application/pdf',
      'text/plain',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    
    if (allowedMimes.includes(file.mimetype) || file.originalname.endsWith('.txt') || file.originalname.endsWith('.pdf')) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PDF and text files are allowed.'));
    }
  }
});
// CORS configuration - allow localhost and kavleen.in
const allowedOrigins = [
  /^http:\/\/localhost(:\d+)?$/,
  /^https?:\/\/.*\.kavleen\.in$/,
  /^https?:\/\/kavleen\.in$/
];

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);
    
    // Check if origin matches any allowed pattern
    const isAllowed = allowedOrigins.some(pattern => pattern.test(origin));
    if (isAllowed) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check endpoint for Cloud Run
app.get('/health', (_req: Request, res: Response) => {
  res.status(200).json({ status: 'healthy', timestamp: new Date().toISOString() });
});

/**
 * Process documents and wait for indexing
 */
async function processDocuments(
  resumeDocId: string,
  jdDocId: string,
  resumeFile: Express.Multer.File,
  jdFile: Express.Multer.File,
  resumeNeedsProcessing: boolean,
  jdNeedsProcessing: boolean
): Promise<void> {
  if (!worker) {
    throw new Error('Document processing worker not initialized');
  }

  const processingPromises: Promise<void>[] = [];

  if (resumeNeedsProcessing) {
    processingPromises.push(worker.processDocument({
      docId: resumeDocId,
      type: 'resume',
      path: resumeFile.originalname,
      buffer: resumeFile.buffer,
      mimetype: resumeFile.mimetype,
      filename: resumeFile.originalname
    }));
  }

  if (jdNeedsProcessing) {
    processingPromises.push(worker.processDocument({
      docId: jdDocId,
      type: 'job_description',
      path: jdFile.originalname,
      buffer: jdFile.buffer,
      mimetype: jdFile.mimetype,
      filename: jdFile.originalname
    }));
  }

  if (processingPromises.length > 0) {
    await Promise.all(processingPromises);
    console.log(`[Server] Document processing completed`);
  }
}

/**
 * Wait for documents to be indexed with timeout
 */
async function waitForIndexing(
  resumeDocId: string,
  jdDocId: string,
  maxWaitTime: number = 600000
): Promise<{ resumeDoc: any; jdDoc: any }> {
  const pollInterval = 1000;
  const startTime = Date.now();

  let resumeDoc = await db.getDocument(resumeDocId);
  let jdDoc = await db.getDocument(jdDocId);

  while ((!resumeDoc || resumeDoc.status !== 'indexed' || !jdDoc || jdDoc.status !== 'indexed') &&
         (Date.now() - startTime) < maxWaitTime) {
    await new Promise(resolve => setTimeout(resolve, pollInterval));
    resumeDoc = await db.getDocument(resumeDocId);
    jdDoc = await db.getDocument(jdDocId);
  }

  return { resumeDoc, jdDoc };
}

/**
 * Calculate match and update comparison
 */
async function calculateAndUpdateMatch(
  comparisonId: string,
  resumeDocId: string,
  jdDocId: string
): Promise<void> {
  if (!matchingService) {
    throw new Error('Matching service not initialized');
  }

  const matchResult = await matchingService.calculateMatch(resumeDocId, jdDocId);
  console.log(`[Server] Match calculation completed: ${matchResult.finalPercent}% match`);

  await db.updateComparisonWithMatch(comparisonId, {
    finalPercent: matchResult.finalPercent,
    semanticScore: matchResult.semanticScore,
    keywordScore: matchResult.keywordScore,
    yearsScore: matchResult.yearsScore,
    resumeYears: matchResult.resumeYears ?? null,
    jdRequiredYears: matchResult.jdRequiredYears ?? null,
    matchedSkills: matchResult.matchedSkills,
    missingSkills: matchResult.missingSkills
  });
}

async function processComparisonInBackground(
  comparisonId: string,
  resumeDocId: string,
  jdDocId: string,
  resumeFile: Express.Multer.File,
  jdFile: Express.Multer.File
): Promise<void> {
  console.log(`[Server] Starting background processing for comparison: ${comparisonId}`);

  try {
    if (!worker) {
      console.error(`[Server] ERROR: Document processing worker not initialized`);
      await db.updateComparisonStatus(comparisonId, 'failed', 'Document processing worker not initialized');
      return;
    }

    // Calculate file hashes and check for reuse
    const resumeHash = DocumentReuseService.calculateFileHash(resumeFile.buffer);
    const jdHash = DocumentReuseService.calculateFileHash(jdFile.buffer);

    const [resumeReuse, jdReuse] = await Promise.all([
      DocumentReuseService.checkDocumentReuse(resumeDocId, resumeHash, 'resume'),
      DocumentReuseService.checkDocumentReuse(jdDocId, jdHash, 'job_description')
    ]);

    // Create documents if needed
    if (resumeReuse.needsProcessing) {
      await db.createDocument(resumeDocId, 'resume', resumeFile.originalname, resumeHash);
    }
    if (jdReuse.needsProcessing) {
      await db.createDocument(jdDocId, 'job_description', jdFile.originalname, jdHash);
    }

    // Update comparison if docIds changed
    if (resumeReuse.finalDocId !== resumeDocId || jdReuse.finalDocId !== jdDocId) {
      await db.updateComparison(comparisonId, {
        resumeDocId: resumeReuse.finalDocId,
        jdDocId: jdReuse.finalDocId
      });
    }

    // Process documents
    await processDocuments(
      resumeDocId,
      jdDocId,
      resumeFile,
      jdFile,
      resumeReuse.needsProcessing,
      jdReuse.needsProcessing
    );

    // Wait for indexing
    const { resumeDoc, jdDoc } = await waitForIndexing(
      resumeReuse.finalDocId,
      jdReuse.finalDocId
    );

    // Calculate match if both indexed
    if (resumeDoc?.status === 'indexed' && jdDoc?.status === 'indexed') {
      console.log(`[Server] Both documents indexed, calculating match...`);
      await calculateAndUpdateMatch(comparisonId, resumeReuse.finalDocId, jdReuse.finalDocId);
    } else {
      const errorMsg = `Document processing failed. Resume: ${resumeDoc?.status}, JD: ${jdDoc?.status}`;
      console.error(`[Server] ERROR: ${errorMsg}`);
      await db.updateComparisonStatus(comparisonId, 'failed', errorMsg);
    }
  } catch (error) {
    console.error(`[Server] ERROR: Background processing failed:`, error);
    await db.updateComparisonStatus(
      comparisonId,
      'failed',
      `Processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

app.post('/api/upload', (req: Request, res: Response, next) => {
  upload.any()(req, res, (err) => {
    if (err) {
      res.status(400).json({
        error: 'Upload error',
        message: err.message
      });
      return;
    }
    next();
  });
}, async (req: Request, res: Response) => {
  try {
    const files = req.files as Express.Multer.File[] | undefined;
    
    if (!files || files.length === 0) {
      console.warn(`[Server] Upload request rejected: No files provided`);
      return res.status(400).json({
        error: 'No file uploaded',
        message: 'Please provide files to upload'
      });
    }

    const resumeFile = files.find(f => f.fieldname === 'resume');
    const jdFile = files.find(f => f.fieldname === 'jobDescription');

    if (!resumeFile || !jdFile) {
      console.warn(`[Server] Upload request rejected: Missing required files`);
      return res.status(400).json({
        error: 'Missing files',
        message: 'Both resume and jobDescription files are required'
      });
    }

    const resumeDocId = `resume:${uuidv4()}`;
    const jdDocId = `jd:${uuidv4()}`;
    const comparisonId = `comparison:${uuidv4()}`;

    let resumeGcsResult = null;
    let jdGcsResult = null;

    if (gcsService) {
      try {
        [resumeGcsResult, jdGcsResult] = await Promise.all([
          gcsService.uploadFile(
            resumeFile.buffer,
            resumeFile.originalname,
            resumeFile.mimetype
          ),
          gcsService.uploadFile(
            jdFile.buffer,
            jdFile.originalname,
            jdFile.mimetype
          )
        ]);
        await db.createFileRecord(
          resumeDocId,
          'resume',
          resumeGcsResult.originalName,
          resumeGcsResult.filename,
          resumeGcsResult.url,
          resumeGcsResult.longTermUrl,
          resumeGcsResult.size,
          resumeGcsResult.mimetype,
          resumeGcsResult.expiresAt,
          resumeGcsResult.longTermExpiresAt
        );

        await db.createFileRecord(
          jdDocId,
          'job_description',
          jdGcsResult.originalName,
          jdGcsResult.filename,
          jdGcsResult.url,
          jdGcsResult.longTermUrl,
          jdGcsResult.size,
          jdGcsResult.mimetype,
          jdGcsResult.expiresAt,
          jdGcsResult.longTermExpiresAt
        );
      } catch (gcsError) {
        console.warn(`[Server] GCS upload failed (continuing without GCS):`, gcsError);
      }
    }

    await db.createComparison(
      comparisonId,
      resumeDocId,
      jdDocId,
      resumeGcsResult?.filename,
      jdGcsResult?.filename,
      resumeGcsResult?.url,
      resumeGcsResult?.longTermUrl,
      jdGcsResult?.url,
      jdGcsResult?.longTermUrl
    );

    res.status(202).json({
      message: 'Files uploaded successfully. Processing started.',
      comparisonId: comparisonId,
      status: 'in_progress'
    });
    processComparisonInBackground(comparisonId, resumeDocId, jdDocId, resumeFile, jdFile)
      .catch(err => {
        console.error(`[Server] Background processing error (already handled):`, err);
      });

    return;

  } catch (error) {
    const err = error as Error;
    return res.status(500).json({
      error: 'Upload failed',
      message: err.message
    });
  }
});

app.get('/api/comparison/:id', async (req: Request, res: Response) => {
  try {
    const comparisonId = req.params.id;

    if (!comparisonId) {
      console.warn(`[Server] Comparison request rejected: Missing comparison ID`);
      return res.status(400).json({
        error: 'Missing comparison ID',
        message: 'Please provide a comparison ID'
      });
    }

    const comparison = await db.getComparison(comparisonId);

    if (!comparison) {
      console.warn(`[Server] Comparison not found: ${comparisonId}`);
      return res.status(404).json({
        error: 'Comparison not found',
        message: `Comparison with ID '${comparisonId}' does not exist`
      });
    }
    return res.json({
      success: true,
      comparison: comparison
    });
  } catch (error) {
    const err = error as Error;
    console.error(`[Server] ERROR: Failed to retrieve comparison:`, err);
    return res.status(500).json({
      error: 'Failed to retrieve comparison',
      message: err.message
    });
  }
});

app.post('/api/comparison/:id/chat', async (req: Request, res: Response) => {
  try {
    const comparisonId = req.params.id;
    const { question } = req.body;
    
    if (!chatService) {
      console.warn(`[Server] Chat request rejected: Chat service not initialized`);
      return res.status(503).json({
        error: 'Chat service not available',
        message: 'Chat service is not initialized'
      });
    }

    if (!comparisonId) {
      console.warn(`[Server] Chat request rejected: Missing comparison ID`);
      return res.status(400).json({
        error: 'Missing comparison ID',
        message: 'Please provide a comparison ID'
      });
    }

    if (!question || typeof question !== 'string') {
      console.warn(`[Server] Chat request rejected: Missing or invalid question`);
      return res.status(400).json({
        error: 'Missing question',
        message: 'Please provide a question in the request body'
      });
    }

    const chatResponse = await chatService.chat(comparisonId, question);

    return res.json({
      success: true,
      answer: chatResponse.answer,
    });
  } catch (error) {
    const err = error as Error;
    console.error(`[Server] ERROR: Chat failed:`, err);
    return res.status(500).json({
      error: 'Chat failed',
      message: err.message
    });
  }
});

const server = app.listen(PORT, () => {
  console.log(`[Server] Server started on port ${PORT}`);
  console.log(`[Server] Health check available at http://localhost:${PORT}/health`);
});

process.on('SIGTERM', async () => {
  server.close(async () => {
    await db.disconnect();
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  server.close(async () => {
    await db.disconnect();
    process.exit(0);
  });
});

