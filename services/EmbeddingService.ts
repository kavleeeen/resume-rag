import { GoogleGenerativeAI } from '@google/generative-ai';

export class EmbeddingService {
  private genAI: GoogleGenerativeAI;
  private model: string;
  private configuredDimension: number;
  private actualDimension: number | null = null;
  private dimensionWarned: boolean = false;

  constructor(apiKey?: string, dimension: number = 1024) {
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY is required for embedding generation');
    }
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = process.env.EMBEDDING_MODEL || 'text-embedding-004';
    this.configuredDimension = dimension;
  }

  async embedText(text: string): Promise<number[]> {
    try {
      const model = this.genAI.getGenerativeModel({ model: this.model });
      const result = await model.embedContent(text);
      let embedding: number[];
      if (result.embedding && result.embedding.values) {
        embedding = result.embedding.values;
      } else if (Array.isArray(result.embedding)) {
        embedding = result.embedding;
      } else {
        throw new Error('Unexpected embedding response format');
      }

      if (this.actualDimension === null) {
        this.actualDimension = embedding.length;
        
        if (this.actualDimension !== this.configuredDimension && !this.dimensionWarned) {
          this.dimensionWarned = true;
        }
      }

      if (embedding.length !== this.configuredDimension) {
        embedding = this.adjustDimension(embedding, this.configuredDimension);
      }

      return embedding;
    } catch (error) {
      throw new Error(`Failed to generate embedding: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private adjustDimension(embedding: number[], targetDimension: number): number[] {
    if (embedding.length === targetDimension) {
      return embedding;
    }

    if (embedding.length > targetDimension) {
      return embedding.slice(0, targetDimension);
    } else {
      const padding = new Array(targetDimension - embedding.length).fill(0);
      return [...embedding, ...padding];
    }
  }

  async embedTexts(texts: string[], batchSize: number = 10): Promise<number[][]> {
    const embeddings: number[][] = [];
    
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchPromises = batch.map(text => this.embedText(text));
      const batchEmbeddings = await Promise.all(batchPromises);
      embeddings.push(...batchEmbeddings);
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

