import { SectionDetectionService } from './SectionDetectionService';

export interface Chunk {
  index: number;
  text: string;
  section?: string;
}

export class ChunkingService {
  private sectionDetector: SectionDetectionService;

  constructor() {
    this.sectionDetector = new SectionDetectionService();
  }

  chunkText(text: string, maxChars: number = 3000, overlapChars: number = 800): Chunk[] {
    if (text.length <= maxChars) {
      const section = this.sectionDetector.getSectionForText(text, 0);
      return [{ index: 0, text, section }];
    }

    const chunks: Chunk[] = [];
    let startIndex = 0;
    let chunkIndex = 0;

    while (startIndex < text.length) {
      let endIndex = Math.min(startIndex + maxChars, text.length);

      if (endIndex < text.length) {
        const paragraphBreak = text.lastIndexOf('\n\n', endIndex);
        if (paragraphBreak > startIndex + maxChars * 0.5) {
          endIndex = paragraphBreak + 2;
        } else {
          const sentenceBreak = text.lastIndexOf('. ', endIndex);
          if (sentenceBreak > startIndex + maxChars * 0.5) {
            endIndex = sentenceBreak + 2;
          } else {
            const lineBreak = text.lastIndexOf('\n', endIndex);
            if (lineBreak > startIndex + maxChars * 0.5) {
              endIndex = lineBreak + 1;
            }
          }
        }
      }

      const chunkText = text.substring(startIndex, endIndex).trim();
      if (chunkText.length > 0) {
        const section = this.sectionDetector.getSectionForText(text, startIndex);
        chunks.push({
          index: chunkIndex,
          text: chunkText,
          section
        });
        chunkIndex++;
      }

      startIndex = Math.max(startIndex + 1, endIndex - overlapChars);
    }

    return chunks;
  }

  deduplicateChunks(chunks: Chunk[]): Chunk[] {
    const seen = new Set<string>();
    const unique: Chunk[] = [];

    for (const chunk of chunks) {
      const normalized = chunk.text
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();

      if (seen.has(normalized)) {
        continue;
      }

      if (normalized.length < 50) {
        continue;
      }

      const words = normalized.split(/\s+/);
      const uniqueWords = new Set(words);
      if (words.length > 10 && uniqueWords.size / words.length < 0.3) {
        continue;
      }

      seen.add(normalized);
      unique.push(chunk);
    }

    return unique;
  }
}

