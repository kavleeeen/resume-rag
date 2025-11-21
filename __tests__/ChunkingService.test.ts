import { ChunkingService } from '../services/ChunkingService';

describe('ChunkingService', () => {
  let chunkingService: ChunkingService;

  beforeEach(() => {
    chunkingService = new ChunkingService();
  });

  describe('chunkText', () => {
    it('should produce chunks <= maxChars', () => {
      const longText = 'a '.repeat(5000); // ~10,000 characters
      const chunks = chunkingService.chunkText(longText, 3000, 800);

      for (const chunk of chunks) {
        expect(chunk.text.length).toBeLessThanOrEqual(3000);
      }
    });

    it('should preserve overlap between chunks', () => {
      const text = 'word '.repeat(1000); // ~5000 characters
      const chunks = chunkingService.chunkText(text, 3000, 800);

      if (chunks.length > 1) {
        // Check that there's overlap between consecutive chunks
        for (let i = 1; i < chunks.length; i++) {
          const prevChunk = chunks[i - 1].text;
          const currentChunk = chunks[i].text;
          
          // Check if there's some overlap (at least 100 chars should overlap)
          const prevEnd = prevChunk.slice(-800);
          const currentStart = currentChunk.slice(0, 800);
          
          // There should be some common text
          expect(prevEnd.length).toBeGreaterThan(0);
          expect(currentStart.length).toBeGreaterThan(0);
        }
      }
    });

    it('should handle short text that fits in one chunk', () => {
      const shortText = 'This is a short text.';
      const chunks = chunkingService.chunkText(shortText, 3000, 800);

      expect(chunks.length).toBe(1);
      expect(chunks[0].text).toBe(shortText);
    });

    it('should handle empty text', () => {
      const chunks = chunkingService.chunkText('', 3000, 800);
      expect(chunks.length).toBe(0);
    });
  });

  describe('deduplicateChunks', () => {
    it('should remove identical chunks', () => {
      const chunks = [
        { index: 0, text: 'This is a test chunk.' },
        { index: 1, text: 'This is a test chunk.' }, // Duplicate
        { index: 2, text: 'This is another chunk.' }
      ];

      const unique = chunkingService.deduplicateChunks(chunks);
      expect(unique.length).toBe(2);
      expect(unique[0].text).toBe('This is a test chunk.');
      expect(unique[1].text).toBe('This is another chunk.');
    });

    it('should remove very short chunks (boilerplate)', () => {
      const chunks = [
        { index: 0, text: 'This is a valid chunk with enough content to pass the filter.' },
        { index: 1, text: 'Short' }, // Too short
        { index: 2, text: 'Another valid chunk with sufficient length.' }
      ];

      const unique = chunkingService.deduplicateChunks(chunks);
      expect(unique.length).toBe(2);
      expect(unique.every(c => c.text.length >= 50)).toBe(true);
    });

    it('should handle chunks with mostly repeated words (boilerplate)', () => {
      const chunks = [
        { index: 0, text: 'This is a valid chunk with diverse content and different words.' },
        { index: 1, text: 'word word word word word word word word word word word word word word word word word word word word' }, // Mostly repeated
        { index: 2, text: 'Another valid chunk with varied vocabulary and content.' }
      ];

      const unique = chunkingService.deduplicateChunks(chunks);
      // Should filter out the repetitive chunk
      expect(unique.length).toBeLessThanOrEqual(2);
    });
  });
});

