import { db } from './DatabaseService';
import { TextExtractionService } from './TextExtractionService';
import { TextCleaningService } from './TextCleaningService';
import { ChunkingService } from './ChunkingService';
import { EmbeddingService } from './EmbeddingService';
import { PineconeService } from './PineconeService';
import { JDSkillExtractionService } from './JDSkillExtractionService';
import { ChunkMetadata } from '../types';
import { createHash } from 'crypto';

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

  private calculateFileHash(buffer: Buffer): string {
    return createHash('sha256').update(buffer).digest('hex');
  }

  async processDocument(params: ProcessDocumentParams): Promise<void> {
    const { docId, type, buffer, mimetype, filename } = params;
    console.log(`[DocumentProcessingWorker] Starting document processing - Doc ID: ${docId}, Type: ${type}`);
    
    // Calculate file hash
    const fileHash = this.calculateFileHash(buffer);
    
    // Check if this file was processed before
    const existingHashRecord = await db.getDocumentByHash(fileHash);
    if (existingHashRecord) {
      const existingDoc = await db.getDocument(existingHashRecord.docId);
      
      if (existingDoc && existingDoc.status === 'indexed') {
        // Check if vectors exist in Pinecone by trying to fetch docvec
        // Use canonical docId (docId is already canonical)
        const docvecId = `${existingHashRecord.docId}::docvec`;
        try {
          const existingDocvec = await this.pineconeService.fetchById(docvecId);
          if (existingDocvec && existingDocvec.values && existingDocvec.values.length > 0) {
            // Copy vectors from existing document to new docId
            await this.copyVectorsFromExisting(existingHashRecord.docId, docId, type, existingDoc.vectorCount || 0);
            
            // Update document status and hash
            await db.updateDocument(docId, {
              status: 'indexed',
              indexedAt: new Date(),
              vectorCount: existingDoc.vectorCount,
              fileHash: fileHash,
              rawText: existingDoc.rawText,
              extractedLength: existingDoc.extractedLength
            });
            
            // Copy JD record if it's a job description
            if (type === 'job_description') {
              const existingJD = await db.getJD(existingHashRecord.docId);
              if (existingJD) {
                await db.createJD(docId, existingJD.topSkills, existingJD.generalSkills, existingJD.requiredYears);
                if (existingJD.skillSynonyms) {
                  await db.updateJD(docId, { skillSynonyms: existingJD.skillSynonyms });
                }
                if (existingJD.skillEmbeddings) {
                  await db.updateJD(docId, { skillEmbeddings: existingJD.skillEmbeddings });
                }
              }
            }
            
            console.log(`[DocumentProcessingWorker] Successfully reused existing document`);
            return;
          }
        } catch (error) {
          // Vectors not found, will process normally
        }
      }
    }
    
    // Store hash record for future lookups
    await db.createFileHashRecord(fileHash, docId, type);
    
    let retries = 0;

    while (retries < MAX_RETRIES) {
      try {
        if (retries > 0) {
          console.log(`[DocumentProcessingWorker] Retry attempt ${retries}/${MAX_RETRIES} for ${docId}`);
        }
        
        await db.setStatus(docId, 'processing');

        // Update document with hash
        await db.updateDocument(docId, { fileHash });

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
        const chunkEmbeddings = await this.embeddingService.embedTexts(chunkTexts, 0);

        const actualDimension = this.embeddingService.getDimension();
        if (actualDimension !== chunkEmbeddings[0]?.length) {
          const errorMsg = `Embedding dimension mismatch: service reports ${actualDimension} but first embedding has ${chunkEmbeddings[0]?.length}`;
          console.error(`[DocumentProcessingWorker] ERROR: ${errorMsg}`);
          throw new Error(errorMsg);
        }

        const docVector = this.embeddingService.meanPool(chunkEmbeddings);

        const uploadedAt = new Date().toISOString();
        const version = 'v1';
        // Use canonical docId (docId is already the canonical ID from DB)
        const canonicalDocId = docId; // docId is the canonical identifier
        const chunkVectors = uniqueChunks.map((chunk, idx) => ({
          id: `${canonicalDocId}::chunk::${chunk.index}`,
          vector: chunkEmbeddings[idx],
          metadata: {
            doc_id: canonicalDocId,
            doc_type: type,
            chunk_index: chunk.index,
            section: chunk.section,
            text_snippet: chunk.text.substring(0, 300),
            version: version,
            uploaded_at: uploadedAt,
            full_text_length: cleanedText.length
          } as ChunkMetadata
        }));

        const docVectorEntry = {
          id: `${canonicalDocId}::docvec`,
          vector: docVector,
          metadata: {
            doc_id: canonicalDocId,
            doc_type: type,
            chunk_index: -1,
            version: version,
            uploaded_at: uploadedAt,
            full_text_length: cleanedText.length
          } as ChunkMetadata
        };

        const totalVectors = chunkVectors.length + 1;
        const allVectors = [...chunkVectors, docVectorEntry];
        await this.pineconeService.upsertChunksBatched(allVectors, 50);

        await db.setIndexed(docId, totalVectors);
        
        // Ensure hash is stored
        await db.updateDocument(docId, { fileHash });
        console.log(`[DocumentProcessingWorker] Document processing completed successfully - ${docId}`);

        return;
      } catch (error) {
        retries++;
        const err = error as Error;
        console.error(`[DocumentProcessingWorker] ERROR processing document ${docId} (attempt ${retries}/${MAX_RETRIES}):`, err);

        if (retries >= MAX_RETRIES) {
          console.error(`[DocumentProcessingWorker] Max retries reached for ${docId}, marking as failed`);
          await db.setStatus(docId, 'failed');
          throw new Error(`Failed to process document after ${MAX_RETRIES} attempts: ${err.message}`);
        }

        const delay = RETRY_DELAY_MS * retries;
        console.log(`[DocumentProcessingWorker] Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  private async copyVectorsFromExisting(
    existingDocId: string,
    newDocId: string,
    type: 'resume' | 'job_description',
    vectorCount: number
  ): Promise<void> {
    try {
      // Use canonical docId (existingDocId is already canonical)
      const docvecId = `${existingDocId}::docvec`;
      const expectedChunkCount = vectorCount - 1; // Subtract 1 for docvec
      
      // For large documents (>200 chunks), use query-based approach to avoid URL length issues
      // For smaller documents, fetch by ID is faster
      const USE_QUERY_APPROACH = expectedChunkCount > 200;
      
      let existingVectors: Array<{ id: string; values: number[]; metadata: ChunkMetadata }> = [];
      
      if (USE_QUERY_APPROACH) {
        // Strategy: Query-based approach for large documents
        // First, get the docvec
        const docvec = await this.pineconeService.fetchById(docvecId);
        if (!docvec || !docvec.values) {
          throw new Error('Docvec not found for existing document');
        }
        
        // Query with filter to get all chunks (with vector values included)
        const queryFilter = {
          doc_id: existingDocId,
          doc_type: type,
          chunk_index: { $gte: 0 }
        };
        
        const queriedChunks = await this.pineconeService.query(docvec.values, Math.max(expectedChunkCount * 2, 5000), queryFilter, true);
        
        // Filter to only chunks from this document
        const validChunks = queriedChunks.filter(chunk => 
          chunk.metadata?.doc_id === existingDocId && 
          chunk.metadata?.chunk_index !== undefined &&
          chunk.metadata?.chunk_index >= 0 &&
          chunk.values && chunk.values.length > 0
        );
        
        // Convert query results to vector format
        const chunkVectors = validChunks.map(chunk => ({
          id: chunk.id,
          values: chunk.values!,
          metadata: chunk.metadata
        }));
        
        existingVectors.push(...chunkVectors);
        existingVectors.push(docvec); // Add docvec
      } else {
        // Strategy: Fetch by ID for smaller documents (faster)
        const chunkIds: string[] = [];
        for (let i = 0; i < expectedChunkCount; i++) {
          chunkIds.push(`${existingDocId}::chunk::${i}`);
        }
        chunkIds.push(docvecId);
        
        existingVectors = await this.pineconeService.fetchByIds(chunkIds);
        
        // Filter out any null/undefined results
        existingVectors = existingVectors.filter(v => v && v.values && v.values.length > 0);
        
        // If we didn't get all chunks, fall back to query approach
        if (existingVectors.length < expectedChunkCount + 1) {
          // Get docvec for query
          const docvec = existingVectors.find(v => v.id === docvecId) || await this.pineconeService.fetchById(docvecId);
          if (!docvec || !docvec.values) {
            throw new Error('Docvec not found for existing document');
          }
          
          // Query with filter to get all chunks (with vector values included)
          const queryFilter = {
            doc_id: existingDocId,
            doc_type: type,
            chunk_index: { $gte: 0 }
          };
          
          const queriedChunks = await this.pineconeService.query(docvec.values, Math.max(expectedChunkCount * 2, 5000), queryFilter, true);
          
          // Filter to only chunks from this document
          const validChunks = queriedChunks.filter(chunk => 
            chunk.metadata?.doc_id === existingDocId && 
            chunk.metadata?.chunk_index !== undefined &&
            chunk.metadata?.chunk_index >= 0 &&
            chunk.values && chunk.values.length > 0
          );
          
          // Merge: use fetched by ID first, then add any missing from query
          const fetchedIds = new Set(existingVectors.map(v => v.id));
          for (const chunk of validChunks) {
            const chunkId = `${existingDocId}::chunk::${chunk.metadata?.chunk_index}`;
            if (!fetchedIds.has(chunkId) && chunk.values) {
              // Add the chunk with its vector values
              existingVectors.push({
                id: chunkId,
                values: chunk.values,
                metadata: chunk.metadata
              });
            }
          }
        }
      }
      
      if (existingVectors.length === 0) {
        throw new Error('No vectors found for existing document');
      }
      
      // Separate chunks and docvec
      const chunks = existingVectors.filter(v => !v.id.endsWith('::docvec'));
      const docvec = existingVectors.find(v => v.id.endsWith('::docvec'));
      
      if (!docvec) {
        throw new Error('Docvec not found in fetched vectors');
      }
      
      // Create new vectors with new docId - preserve ALL metadata fields exactly
      // Use canonical docId (newDocId is already canonical)
      const uploadedAt = new Date().toISOString();
      const version = 'v1';
      const newVectors = chunks.map(chunk => {
        const chunkIndex = chunk.metadata?.chunk_index ?? parseInt(chunk.id.split('::').pop() || '0', 10);
        const newId = `${newDocId}::chunk::${chunkIndex}`;
        
        // Preserve ALL original metadata, just update doc_id, version, and uploaded_at
        return {
          id: newId,
          vector: chunk.values,
          metadata: {
            doc_id: newDocId,
            doc_type: type,
            chunk_index: chunkIndex,
            version: version,
            uploaded_at: uploadedAt,
            ...(chunk.metadata?.section !== undefined && { section: chunk.metadata.section }),
            ...(chunk.metadata?.text_snippet !== undefined && { text_snippet: chunk.metadata.text_snippet }),
            ...(chunk.metadata?.full_text_length !== undefined && { full_text_length: chunk.metadata.full_text_length }),
            ...(chunk.metadata?.source !== undefined && { source: chunk.metadata.source }),
            ...(chunk.metadata?.filename !== undefined && { filename: chunk.metadata.filename }),
            ...(chunk.metadata?.mimetype !== undefined && { mimetype: chunk.metadata.mimetype }),
            ...(chunk.metadata?.pages !== undefined && { pages: chunk.metadata.pages }),
            ...(chunk.metadata?.estimated_tokens !== undefined && { estimated_tokens: chunk.metadata.estimated_tokens }),
            ...(chunk.metadata?.has_overlap !== undefined && { has_overlap: chunk.metadata.has_overlap }),
            ...(chunk.metadata?.overlap_size !== undefined && { overlap_size: chunk.metadata.overlap_size }),
            ...(chunk.metadata?.embedding_model !== undefined && { embedding_model: chunk.metadata.embedding_model }),
            ...(chunk.metadata?.embedding_dimension !== undefined && { embedding_dimension: chunk.metadata.embedding_dimension })
          } as ChunkMetadata
        };
      });
      
      // Add the docvec with all metadata preserved
      newVectors.push({
        id: `${newDocId}::docvec`,
        vector: docvec.values,
        metadata: {
          doc_id: newDocId,
          doc_type: type,
          chunk_index: -1,
          version: version,
          uploaded_at: uploadedAt,
          ...(docvec.metadata?.section !== undefined && { section: docvec.metadata.section }),
          ...(docvec.metadata?.text_snippet !== undefined && { text_snippet: docvec.metadata.text_snippet }),
          ...(docvec.metadata?.full_text_length !== undefined && { full_text_length: docvec.metadata.full_text_length }),
          ...(docvec.metadata?.source !== undefined && { source: docvec.metadata.source }),
          ...(docvec.metadata?.filename !== undefined && { filename: docvec.metadata.filename }),
          ...(docvec.metadata?.mimetype !== undefined && { mimetype: docvec.metadata.mimetype }),
          ...(docvec.metadata?.pages !== undefined && { pages: docvec.metadata.pages }),
          ...(docvec.metadata?.estimated_tokens !== undefined && { estimated_tokens: docvec.metadata.estimated_tokens }),
          ...(docvec.metadata?.has_overlap !== undefined && { has_overlap: docvec.metadata.has_overlap }),
          ...(docvec.metadata?.overlap_size !== undefined && { overlap_size: docvec.metadata.overlap_size }),
          ...(docvec.metadata?.embedding_model !== undefined && { embedding_model: docvec.metadata.embedding_model }),
          ...(docvec.metadata?.embedding_dimension !== undefined && { embedding_dimension: docvec.metadata.embedding_dimension })
        } as ChunkMetadata
      });
      
      await this.pineconeService.upsertChunksBatched(newVectors, 50);
    } catch (error) {
      console.error(`[DocumentProcessingWorker] ERROR copying vectors:`, error);
      throw error;
    }
  }
}

