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
import { v4 as uuidv4 } from 'uuid';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5003;

const geminiApiKey = process.env.GEMINI_API_KEY;
const pineconeApiKey = process.env.PINECONE_API_KEY;
const pineconeIndexName = process.env.PINECONE_INDEX_NAME || 'resume-rag-index';
const embeddingDimension = parseInt(process.env.EMBEDDING_DIMENSION || '768', 10);

if (!geminiApiKey) {
  throw new Error('GEMINI_API_KEY environment variable is required');
}
if (!pineconeApiKey) {
  throw new Error('PINECONE_API_KEY environment variable is required');
}

db.connect().catch(err => {
  process.exit(1);
});
const embeddingService = new EmbeddingService(geminiApiKey, embeddingDimension);
const pineconeService = new PineconeService(pineconeApiKey, pineconeIndexName, embeddingDimension);
const matchingService = new MatchingService(pineconeService, embeddingService);

const worker = new DocumentProcessingWorker(
  geminiApiKey,
  pineconeApiKey,
  pineconeIndexName,
  embeddingDimension
);

let gcsService: GCStorageService | null = null;
try {
  gcsService = new GCStorageService();
} catch (error) {
}

const chatService = new ChatService(geminiApiKey, embeddingService, pineconeService);

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
app.use(cors({
  origin: /^http:\/\/localhost(:\d+)?$/,
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

async function processComparisonInBackground(
  comparisonId: string,
  resumeDocId: string,
  jdDocId: string,
  resumeFile: Express.Multer.File,
  jdFile: Express.Multer.File
): Promise<void> {
  try {
    await db.createDocument(resumeDocId, 'resume', resumeFile.originalname);
    await db.createDocument(jdDocId, 'job_description', jdFile.originalname);
    const processResume = worker.processDocument({
      docId: resumeDocId,
      type: 'resume',
      path: resumeFile.originalname,
      buffer: resumeFile.buffer,
      mimetype: resumeFile.mimetype,
      filename: resumeFile.originalname
    });

    const processJD = worker.processDocument({
      docId: jdDocId,
      type: 'job_description',
      path: jdFile.originalname,
      buffer: jdFile.buffer,
      mimetype: jdFile.mimetype,
      filename: jdFile.originalname
    });

    await Promise.all([processResume, processJD]);

    const resumeDoc = await db.getDocument(resumeDocId);
    const jdDoc = await db.getDocument(jdDocId);

    if (resumeDoc?.status === 'indexed' && jdDoc?.status === 'indexed') {
      try {
        const matchResult = await matchingService.calculateMatch(resumeDocId, jdDocId);
        
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
      } catch (matchError) {
        await db.updateComparisonStatus(
          comparisonId,
          'failed',
          `Match calculation failed: ${matchError instanceof Error ? matchError.message : 'Unknown error'}`
        );
      }
    } else {
      const errorMsg = `Document processing failed. Resume: ${resumeDoc?.status}, JD: ${jdDoc?.status}`;
      await db.updateComparisonStatus(comparisonId, 'failed', errorMsg);
    }
  } catch (error) {
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
      return res.status(400).json({
        error: 'No file uploaded',
        message: 'Please provide files to upload'
      });
    }

    const resumeFile = files.find(f => f.fieldname === 'resume');
    const jdFile = files.find(f => f.fieldname === 'jobDescription');

    if (!resumeFile || !jdFile) {
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
        resumeGcsResult = await gcsService.uploadFile(
          resumeFile.buffer,
          resumeFile.originalname,
          resumeFile.mimetype
        );

        jdGcsResult = await gcsService.uploadFile(
          jdFile.buffer,
          jdFile.originalname,
          jdFile.mimetype
        );
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
      return res.status(400).json({
        error: 'Missing comparison ID',
        message: 'Please provide a comparison ID'
      });
    }

    const comparison = await db.getComparison(comparisonId);

    if (!comparison) {
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

    if (!comparisonId) {
      return res.status(400).json({
        error: 'Missing comparison ID',
        message: 'Please provide a comparison ID'
      });
    }

    if (!question || typeof question !== 'string') {
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
    return res.status(500).json({
      error: 'Chat failed',
      message: err.message
    });
  }
});

const server = app.listen(PORT, () => {
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

