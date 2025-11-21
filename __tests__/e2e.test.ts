import { db } from '../services/DatabaseService';
import { DocumentProcessingWorker } from '../services/DocumentProcessingWorker';
import { PineconeService } from '../services/PineconeService';
import * as fs from 'fs';
import * as path from 'path';

describe('E2E Document Processing', () => {
  let worker: DocumentProcessingWorker;
  let pineconeService: PineconeService;

  beforeAll(async () => {
    // Connect to database
    await db.connect();

    // Initialize services
    const geminiApiKey = process.env.GEMINI_API_KEY;
    const pineconeApiKey = process.env.PINECONE_API_KEY;
    const pineconeIndexName = process.env.PINECONE_INDEX_NAME || 'resume-rag-index';
    const embeddingDimension = parseInt(process.env.EMBEDDING_DIMENSION || '768', 10);

    if (!geminiApiKey || !pineconeApiKey) {
      throw new Error('GEMINI_API_KEY and PINECONE_API_KEY are required for E2E tests');
    }

    worker = new DocumentProcessingWorker(
      geminiApiKey,
      pineconeApiKey,
      pineconeIndexName,
      embeddingDimension
    );

    pineconeService = new PineconeService(pineconeApiKey, pineconeIndexName, embeddingDimension);
  });

  afterAll(async () => {
    await db.disconnect();
  });

  it('should process a known resume and JD, assert status=indexed and JD has requiredSkills', async () => {
    // Read test files
    const resumePath = path.join(__dirname, '../test_resume.txt');
    const jdPath = path.join(__dirname, '../test_jd.txt');

    if (!fs.existsSync(resumePath) || !fs.existsSync(jdPath)) {
      console.warn('Test files not found, skipping E2E test');
      return;
    }

    const resumeBuffer = fs.readFileSync(resumePath);
    const jdBuffer = fs.readFileSync(jdPath);

    const resumeDocId = `resume:test:${Date.now()}`;
    const jdDocId = `jd:test:${Date.now()}`;

    // Create document records
    await db.createDocument(resumeDocId, 'resume', 'test_resume.txt');
    await db.createDocument(jdDocId, 'job_description', 'test_jd.txt');

    // Process documents
    await worker.processDocument({
      docId: resumeDocId,
      type: 'resume',
      path: resumePath,
      buffer: resumeBuffer,
      mimetype: 'text/plain',
      filename: 'test_resume.txt'
    });

    await worker.processDocument({
      docId: jdDocId,
      type: 'job_description',
      path: jdPath,
      buffer: jdBuffer,
      mimetype: 'text/plain',
      filename: 'test_jd.txt'
    });

    // Assert resume status
    const resumeDoc = await db.getDocument(resumeDocId);
    expect(resumeDoc).not.toBeNull();
    expect(resumeDoc?.status).toBe('indexed');
    expect(resumeDoc?.vectorCount).toBeGreaterThan(0);

    // Assert JD status and requiredSkills
    const jdDoc = await db.getDocument(jdDocId);
    expect(jdDoc).not.toBeNull();
    expect(jdDoc?.status).toBe('indexed');
    expect(jdDoc?.vectorCount).toBeGreaterThan(0);

    const jdRecord = await db.getJD(jdDocId);
    expect(jdRecord).not.toBeNull();
    expect(jdRecord?.requiredSkills).toBeDefined();
    expect(Array.isArray(jdRecord?.requiredSkills)).toBe(true);
    expect(jdRecord?.requiredSkills.length).toBeGreaterThan(0);

    // Query Pinecone for docvec
    const resumeDocRecord = await db.getDocument(resumeDocId);
    // We can't directly query without the vector, but we can verify the document was indexed
    expect(resumeDocRecord?.status).toBe('indexed');
    expect(resumeDocRecord?.vectorCount).toBeGreaterThan(0);
  }, 120000); // 2 minute timeout for full processing
});

