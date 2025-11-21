import { db } from './DatabaseService';
import { TextExtractionService } from './TextExtractionService';
import { TextCleaningService } from './TextCleaningService';
import { ChunkingService } from './ChunkingService';
import { EmbeddingService } from './EmbeddingService';
import { PineconeService } from './PineconeService';
import { JDSkillExtractionService } from './JDSkillExtractionService';
import { ChunkMetadata } from '../types';

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;

export interface ProcessDocumentParams {
  docId: string;
  type: 'resume' | 'job_description';
  path: string; // File path or buffer reference
  buffer: Buffer;
  mimetype: string;
  filename: string;
}

export class DocumentProcessingWorker {
  private textExtractor: TextExtractionService;
  private textCleaner: TextCleaningService;
  private chunker: ChunkingService;
  private embeddingService: EmbeddingService;
  private pineconeService: PineconeService;
  private jdSkillExtractor: JDSkillExtractionService;

  constructor(
    geminiApiKey: string,
    pineconeApiKey: string,
    pineconeIndexName: string,
    embeddingDimension: number
  ) {
    this.textExtractor = new TextExtractionService();
    this.textCleaner = new TextCleaningService();
    this.chunker = new ChunkingService();
    this.embeddingService = new EmbeddingService(geminiApiKey, embeddingDimension);
    this.pineconeService = new PineconeService(pineconeApiKey, pineconeIndexName, embeddingDimension);
    this.jdSkillExtractor = new JDSkillExtractionService(geminiApiKey);
  }

  async processDocument(params: ProcessDocumentParams): Promise<void> {
    const { docId, type, buffer, mimetype, filename } = params;
    let retries = 0;

    while (retries < MAX_RETRIES) {
      try {
        await db.setStatus(docId, 'processing');

        const extractionResult = await this.textExtractor.extractText(buffer, mimetype, filename);
        const extractedLength = extractionResult.text.length;

        await db.setRawText(docId, extractionResult.text, extractedLength);

        const cleanedText = this.textCleaner.cleanText(extractionResult.text);

        if (type === 'job_description') {
          const skills = await this.jdSkillExtractor.extractSkills(extractionResult.text);
          
          await db.createJD(docId, skills.topSkills, skills.generalSkills, skills.requiredYears);
          
          if (skills.skillSynonyms && Object.keys(skills.skillSynonyms).length > 0) {
            await db.updateJD(docId, { skillSynonyms: skills.skillSynonyms });
          }
        }

        const chunks = this.chunker.chunkText(cleanedText, 3000, 800);

        const uniqueChunks = this.chunker.deduplicateChunks(chunks);

        const chunkTexts = uniqueChunks.map(c => c.text);
        const chunkEmbeddings = await this.embeddingService.embedTexts(chunkTexts, 20);

        const actualDimension = this.embeddingService.getDimension();
        if (actualDimension !== chunkEmbeddings[0]?.length) {
          throw new Error(`Embedding dimension mismatch: service reports ${actualDimension} but first embedding has ${chunkEmbeddings[0]?.length}`);
        }

        const docVector = this.embeddingService.meanPool(chunkEmbeddings);

        const chunkVectors = uniqueChunks.map((chunk, idx) => ({
          id: `${docId}:chunk:${chunk.index}`,
          vector: chunkEmbeddings[idx],
          metadata: {
            doc_id: docId,
            doc_type: type,
            chunk_index: chunk.index,
            section: chunk.section,
            text_snippet: chunk.text.substring(0, 200),
            full_text_length: cleanedText.length
          } as ChunkMetadata
        }));

        const docVectorEntry = {
          id: `${docId}:docvec`,
          vector: docVector,
          metadata: {
            doc_id: docId,
            doc_type: type,
            chunk_index: -1,
            full_text_length: cleanedText.length
          } as ChunkMetadata
        };

        await this.pineconeService.upsertChunks([...chunkVectors, docVectorEntry]);
        const totalVectors = chunkVectors.length + 1;

        await db.setIndexed(docId, totalVectors);

        return;
      } catch (error) {
        retries++;
        const err = error as Error;

        if (retries >= MAX_RETRIES) {
          await db.setStatus(docId, 'failed');
          throw new Error(`Failed to process document after ${MAX_RETRIES} attempts: ${err.message}`);
        }

        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS * retries));
      }
    }
  }
}

