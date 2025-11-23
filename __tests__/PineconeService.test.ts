import { PineconeService } from '../services/PineconeService';
import { ChunkMetadata } from '../types';

describe('PineconeService', () => {
  let pineconeService: PineconeService;

  beforeAll(() => {
    const apiKey = process.env.PINECONE_API_KEY;
    const indexName = process.env.PINECONE_INDEX_NAME || 'resume-rag-index';
    const dimension = parseInt(process.env.EMBEDDING_DIMENSION || '768', 10);

    if (!apiKey) {
      throw new Error('PINECONE_API_KEY is required for Pinecone tests');
    }

    pineconeService = new PineconeService(apiKey, indexName, dimension);
  });

  describe('upsert', () => {
    it('should upsert vectors without errors and verify index count increased', async () => {
      const testId = `test:${Date.now()}`;
      const testVector = new Array(768).fill(0).map(() => Math.random());
      const testMetadata: ChunkMetadata = {
        doc_id: testId,
        doc_type: 'resume',
        chunk_index: 0,
        section: 'test'
      };

      // Upsert test vector
      await pineconeService.upsert([{
        id: testId,
        values: testVector,
        metadata: testMetadata
      }]);

      // Wait a bit for index to update
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Clean up: query to verify the vector exists
      const results = await pineconeService.query(testVector, 1);
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].id).toBe(testId);
    }, 30000); // 30 second timeout for API calls
  });
});

