import { db } from './DatabaseService';
import { createHash } from 'crypto';

export interface DocumentReuseResult {
  finalDocId: string;
  needsProcessing: boolean;
}

/**
 * Service for handling document reuse based on file hashes
 * Follows Single Responsibility Principle
 */
export class DocumentReuseService {
  /**
   * Calculate SHA-256 hash of file buffer
   */
  static calculateFileHash(buffer: Buffer): string {
    return createHash('sha256').update(buffer).digest('hex');
  }

  /**
   * Check if document can be reused based on file hash
   * Returns the final docId to use and whether processing is needed
   */
  static async checkDocumentReuse(
    originalDocId: string,
    fileHash: string,
    _type: 'resume' | 'job_description'
  ): Promise<DocumentReuseResult> {
    const existingHashRecord = await db.getDocumentByHash(fileHash);
    
    if (!existingHashRecord) {
      return {
        finalDocId: originalDocId,
        needsProcessing: true
      };
    }

    const existingDoc = await db.getDocument(existingHashRecord.docId);
    
    if (existingDoc && existingDoc.status === 'indexed') {
      return {
        finalDocId: existingHashRecord.docId,
        needsProcessing: false
      };
    }

    return {
      finalDocId: originalDocId,
      needsProcessing: true
    };
  }
}

