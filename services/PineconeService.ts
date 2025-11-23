import { Pinecone } from '@pinecone-database/pinecone';
import { ChunkMetadata } from '../types';

export class PineconeService {
  private index: any; // Pinecone index
  private dimension: number;

  constructor(apiKey: string, indexName: string, dimension: number) {
    const pinecone = new Pinecone({ apiKey });
    this.index = pinecone.index(indexName);
    this.dimension = dimension;
  }

  async upsert(vectors: Array<{ id: string; values: number[]; metadata: ChunkMetadata }>): Promise<void> {
    try {
      // Validate vector dimensions are consistent
      if (vectors.length === 0) {
        console.error(`[PineconeService] ERROR: Cannot upsert empty vector array`);
        throw new Error('Cannot upsert empty vector array');
      }

      const firstDimension = vectors[0].values.length;
      for (const vector of vectors) {
        if (vector.values.length !== firstDimension) {
          const errorMsg = `Vector dimension mismatch: expected ${firstDimension} but found ${vector.values.length}`;
          console.error(`[PineconeService] ERROR: ${errorMsg}`);
          throw new Error(errorMsg);
        }
      }

      if (this.dimension && firstDimension !== this.dimension) {
        console.warn(`[PineconeService] WARNING: Vector dimension (${firstDimension}) doesn't match configured dimension (${this.dimension})`);
      }

      await this.index.upsert(vectors);
    } catch (error) {
      const err = error as Error;
      console.error(`[PineconeService] ERROR: Failed to upsert vectors:`, err);
      if (err.message.includes('dimension')) {
        throw new Error(
          `Failed to upsert vectors to Pinecone: ${err.message}. ` +
          `Your Pinecone index is configured for ${this.dimension} dimensions, but vectors have ${vectors[0]?.values.length || 'unknown'} dimensions. ` +
          `Please recreate your Pinecone index with ${vectors[0]?.values.length || 768} dimensions to match your embedding model.`
        );
      }
      throw new Error(`Failed to upsert vectors to Pinecone: ${err.message}`);
    }
  }

  async upsertChunks(chunks: Array<{ id: string; vector: number[]; metadata: ChunkMetadata }>): Promise<void> {
    const vectors = chunks.map(chunk => ({
      id: chunk.id,
      values: chunk.vector,
      metadata: chunk.metadata
    }));
    await this.upsert(vectors);
  }

  async upsertChunksBatched(chunks: Array<{ id: string; vector: number[]; metadata: ChunkMetadata }>, batchSize: number = 50): Promise<void> {
    if (chunks.length === 0) {
      return;
    }

    const vectors = chunks.map(chunk => ({
      id: chunk.id,
      values: chunk.vector,
      metadata: chunk.metadata
    }));

    for (let i = 0; i < vectors.length; i += batchSize) {
      const batch = vectors.slice(i, i + batchSize);
      await this.upsert(batch);
    }
  }

  async query(vector: number[], topK: number = 5, filter?: any, includeValues: boolean = false, includeMetadata: boolean = true): Promise<Array<{ id: string; score: number; metadata: ChunkMetadata; values?: number[] }>> {
    try {
      const queryResponse = await this.index.query({
        vector,
        topK,
        includeMetadata: includeMetadata,
        includeValues: includeValues,
        filter
      });

      const matches = queryResponse.matches.map((match: any) => ({
        id: match.id,
        score: match.score,
        metadata: match.metadata as ChunkMetadata,
        ...(includeValues && match.values ? { values: match.values } : {})
      }));
      return matches;
    } catch (error) {
      console.error(`[PineconeService] ERROR: Failed to query Pinecone:`, error);
      throw new Error(`Failed to query Pinecone: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async getIndexMetric(): Promise<'cosine' | 'euclidean' | 'dotproduct'> {
    try {
      const stats = await this.index.describeIndexStats();
      return (stats.metric as 'cosine' | 'euclidean' | 'dotproduct') || 'cosine';
    } catch (error) {
      return 'cosine';
    }
  }

  async fetchById(id: string): Promise<{ id: string; values: number[]; metadata: ChunkMetadata } | null> {
    try {
      const fetchResponse = await this.index.fetch([id]);
      const record = fetchResponse.records?.[id];
      if (!record) {
        return null;
      }
      return {
        id: record.id,
        values: record.values,
        metadata: record.metadata as ChunkMetadata
      };
    } catch (error) {
      console.error(`[PineconeService] ERROR: Failed to fetch vector by ID:`, error);
      throw new Error(`Failed to fetch vector by ID from Pinecone: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async fetchByIds(ids: string[]): Promise<Array<{ id: string; values: number[]; metadata: ChunkMetadata }>> {
    try {
      if (ids.length === 0) {
        return [];
      }
      
      // Pinecone has limits on URL length, so batch large requests
      const BATCH_SIZE = 100; // Safe batch size to avoid URL length limits
      const allFetchedVectors: Array<{ id: string; values: number[]; metadata: ChunkMetadata }> = [];
      
      if (ids.length <= BATCH_SIZE) {
        // Small request, fetch all at once
        const fetchResponse = await this.index.fetch(ids);
        const records = fetchResponse.records || {};
        const fetchedVectors = Object.values(records).map((record: any) => ({
          id: record.id,
          values: record.values,
          metadata: record.metadata as ChunkMetadata
        }));
        return fetchedVectors;
      } else {
        // Large request, batch it
        for (let i = 0; i < ids.length; i += BATCH_SIZE) {
          const batch = ids.slice(i, i + BATCH_SIZE);
          
          const fetchResponse = await this.index.fetch(batch);
          const records = fetchResponse.records || {};
          const fetchedVectors = Object.values(records).map((record: any) => ({
            id: record.id,
            values: record.values,
            metadata: record.metadata as ChunkMetadata
          }));
          
          allFetchedVectors.push(...fetchedVectors);
        }
        
        return allFetchedVectors;
      }
    } catch (error) {
      console.error(`[PineconeService] ERROR: Failed to fetch vectors by IDs:`, error);
      throw new Error(`Failed to fetch vectors by IDs from Pinecone: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

