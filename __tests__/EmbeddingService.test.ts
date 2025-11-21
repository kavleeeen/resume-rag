import { EmbeddingService } from '../services/EmbeddingService';

describe('EmbeddingService', () => {
  let embeddingService: EmbeddingService;

  beforeAll(() => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY is required for embedding tests');
    }
    embeddingService = new EmbeddingService(apiKey);
  });

  describe('embedTexts', () => {
    it('should return vectors of consistent dimension', async () => {
      const texts = [
        'This is a test text.',
        'Another test text for embedding.',
        'Yet another text to embed.'
      ];

      const embeddings = await embeddingService.embedTexts(texts, 10);

      expect(embeddings.length).toBe(texts.length);
      
      const dimension = embeddingService.getDimension();
      for (const embedding of embeddings) {
        expect(embedding.length).toBe(dimension);
        expect(Array.isArray(embedding)).toBe(true);
        expect(embedding.every(v => typeof v === 'number')).toBe(true);
      }
    }, 30000); // 30 second timeout for API calls

    it('should handle batch processing', async () => {
      const texts = Array(25).fill(0).map((_, i) => `Test text ${i}`);
      const batchSize = 10;

      const embeddings = await embeddingService.embedTexts(texts, batchSize);

      expect(embeddings.length).toBe(texts.length);
      const dimension = embeddingService.getDimension();
      embeddings.forEach(embedding => {
        expect(embedding.length).toBe(dimension);
      });
    }, 60000); // 60 second timeout for larger batch
  });

  describe('meanPool', () => {
    it('should compute mean of vectors correctly', () => {
      const vectors = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ];

      const mean = embeddingService.meanPool(vectors);

      expect(mean.length).toBe(3);
      expect(mean[0]).toBeCloseTo(4); // (1+4+7)/3
      expect(mean[1]).toBeCloseTo(5); // (2+5+8)/3
      expect(mean[2]).toBeCloseTo(6); // (3+6+9)/3
    });

    it('should throw error for empty vector array', () => {
      expect(() => {
        embeddingService.meanPool([]);
      }).toThrow('Cannot compute mean of empty vector array');
    });

    it('should throw error for vectors with different dimensions', () => {
      const vectors = [
        [1, 2, 3],
        [4, 5] // Different dimension
      ];

      expect(() => {
        embeddingService.meanPool(vectors);
      }).toThrow('All vectors must have the same dimension');
    });
  });
});

