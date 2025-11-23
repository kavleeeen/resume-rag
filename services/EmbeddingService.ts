import { GoogleGenerativeAI } from '@google/generative-ai';

export class EmbeddingService {
  private genAI: GoogleGenerativeAI;
  private model: string;
  private configuredDimension: number;
  private actualDimension: number | null = null;
  private dimensionWarned: boolean = false;
  private readonly MAX_RETRIES = 3;
  private readonly INITIAL_RETRY_DELAY_MS = 1000;
  private embeddingCache: Map<string, number[]> = new Map();
  private modelInstance: any = null; // Cache model instance for reuse

  constructor(apiKey?: string, dimension: number = 1024) {
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY is required for embedding generation');
    }
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = process.env.EMBEDDING_MODEL || 'text-embedding-004';
    this.configuredDimension = dimension;
    // Pre-instantiate model for faster subsequent calls
    this.modelInstance = this.genAI.getGenerativeModel({ model: this.model });
  }

  async embedText(text: string, retryCount: number = 0): Promise<number[]> {
    // Check cache first (useful for duplicate chunks) - removed logging for speed
    const cacheKey = text.substring(0, 1000);
    if (this.embeddingCache.has(cacheKey) && retryCount === 0) {
      return [...this.embeddingCache.get(cacheKey)!]; // Return copy to avoid mutation
    }

    try {
      // Only log retries
      if (retryCount > 0) {
        console.log(`[EmbeddingService] Retry ${retryCount}/${this.MAX_RETRIES}...`);
      }

      // Use cached model instance
      const result = await this.modelInstance.embedContent(text);
      let embedding: number[];
      if (result.embedding && result.embedding.values) {
        embedding = result.embedding.values;
      } else if (Array.isArray(result.embedding)) {
        embedding = result.embedding;
      } else {
        throw new Error('Unexpected embedding response format');
      }

      // Dimension check only once
      if (this.actualDimension === null) {
        this.actualDimension = embedding.length;
        if (this.actualDimension !== this.configuredDimension && !this.dimensionWarned) {
          console.warn(`[EmbeddingService] Dimension mismatch: ${this.configuredDimension} vs ${this.actualDimension}`);
          this.dimensionWarned = true;
        }
      }

      // Adjust dimension if needed (inline for speed)
      if (embedding.length !== this.configuredDimension) {
        if (embedding.length > this.configuredDimension) {
          embedding = embedding.slice(0, this.configuredDimension);
        } else {
          const padding = new Array(this.configuredDimension - embedding.length).fill(0);
          embedding = [...embedding, ...padding];
        }
      }

      // Cache result (silently)
      if (retryCount === 0 && !this.embeddingCache.has(cacheKey)) {
        if (this.embeddingCache.size >= 1000) {
          const firstKey = this.embeddingCache.keys().next().value;
          if (firstKey) this.embeddingCache.delete(firstKey);
        }
        this.embeddingCache.set(cacheKey, embedding);
      }

      return embedding;
    } catch (error) {
      const err = error as Error;
      const isNetworkError = err.message.includes('fetch failed') ||
        err.message.includes('ECONNRESET') ||
        err.message.includes('ETIMEDOUT') ||
        err.message.includes('network');

      if (isNetworkError && retryCount < this.MAX_RETRIES) {
        const delay = this.INITIAL_RETRY_DELAY_MS * Math.pow(2, retryCount); // Exponential backoff
        console.warn(`[EmbeddingService] Network error (attempt ${retryCount + 1}/${this.MAX_RETRIES + 1}), retrying in ${delay}ms:`, err.message);
        await new Promise(resolve => setTimeout(resolve, delay));
        return this.embedText(text, retryCount + 1);
      }

      console.error(`[EmbeddingService] ERROR: Failed to generate embedding after ${retryCount} retries:`, err);
      throw new Error(`Failed to generate embedding: ${err.message}`);
    }
  }


  async embedTexts(texts: string[], _batchSize: number = 10): Promise<number[][]> {
    // Optimized settings - higher defaults for maximum speed
    const maxConcurrent = parseInt(process.env.EMBEDDING_MAX_CONCURRENT || '30', 10); // Increased to 30
    const requestDelay = parseInt(process.env.EMBEDDING_REQUEST_DELAY_MS || '0', 10);

    // Simplified: process all texts with controlled concurrency - no batching overhead
    const results: Array<{ index: number; embedding: number[] | null; error: Error | null }> = [];

    // Process in chunks with maximum parallelism
    for (let i = 0;i < texts.length;i += maxConcurrent) {
      const chunk = texts.slice(i, i + maxConcurrent);
      const chunkPromises = chunk.map(async (text, chunkIdx) => {
        const index = i + chunkIdx;
        try {
          // Minimal delay only if explicitly set (for rate limiting)
          if (requestDelay > 0 && chunkIdx > 0) {
            await new Promise(resolve => setTimeout(resolve, requestDelay * chunkIdx));
          }

          const embedding = await this.embedText(text);
          results.push({ index, embedding, error: null });
        } catch (error) {
          results.push({ index, embedding: null, error: error as Error });
        }
      });

      // Process chunk in parallel - no waiting between chunks
      await Promise.all(chunkPromises);
    }

    // Sort by index and collect embeddings
    results.sort((a, b) => a.index - b.index);
    const embeddings: number[][] = [];
    const errors: Array<{ index: number; error: Error }> = [];

    for (const result of results) {
      if (result.embedding) {
        embeddings.push(result.embedding);
      } else if (result.error) {
        errors.push({ index: result.index, error: result.error });
      }
    }
    
    // Handle errors
    if (errors.length > 0) {
      console.error(`[EmbeddingService] ${errors.length}/${texts.length} embeddings failed`);
      if (errors.length > texts.length * 0.1) {
        throw new Error(`Too many failures: ${errors.length}/${texts.length}`);
      }
    }

    return embeddings;
  }

  getDimension(): number {
    return this.configuredDimension;
  }

  meanPool(vectors: number[][]): number[] {
    if (vectors.length === 0) {
      throw new Error('Cannot compute mean of empty vector array');
    }

    const dimension = vectors[0].length;
    const mean = new Array(dimension).fill(0);

    for (const vector of vectors) {
      if (vector.length !== dimension) {
        throw new Error('All vectors must have the same dimension');
      }
      for (let i = 0; i < dimension; i++) {
        mean[i] += vector[i];
      }
    }

    for (let i = 0; i < dimension; i++) {
      mean[i] /= vectors.length;
    }

    return mean;
  }
}

