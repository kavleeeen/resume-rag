export interface ExtractionResult {
  text: string;
  pages: number;
  metadata: {
    info?: any;
    metadata?: any;
    version?: any;
    encoding?: string;
    lineCount?: number;
  };
}

export interface ChunkMetadata {
  // Document identification
  doc_id: string; // e.g., "resume:1234" or "jd:5678"
  doc_type: 'resume' | 'job_description';
  chunk_index: number;
  
  section?: string;
  
  source?: string;
  filename?: string;
  mimetype?: string;
  pages?: number;
  estimated_tokens?: number;
  has_overlap?: boolean;
  overlap_size?: number;
  text_snippet?: string;
  full_text_length?: number;
  embedding_model?: string;
  embedding_dimension?: number;
}

export interface Chunk {
  text: string;
  metadata: ChunkMetadata;
}

export interface ChunkWithEmbedding {
  id: string;
  vector: number[];
  metadata: ChunkMetadata;
}

export interface ProcessingSummary {
  filename: string;
  source: string;
  originalTextLength: number;
  cleanedTextLength: number;
  numChunks: number;
  pages: number;
  totalTokens: number;
}

import { ResumeAnalysis, JDRequirements } from '../services/ResumeAnalysisService';

export interface ProcessingResult {
  chunks: ChunkWithEmbedding[];
  docId: string; // Document ID for matching
  summary: ProcessingSummary;
  extractedText?: string | null; // Full text
  resumeAnalysis?: ResumeAnalysis | null;
  jdRequirements?: JDRequirements | null;
}

export interface DocumentInput {
  buffer: Buffer;
  mimetype: string;
  filename: string;
  source: 'resume' | 'jobDescription';
}

export interface SearchResult {
  id: string;
  score: number;
  metadata: ChunkMetadata;
}

export interface VectorDBFilter {
  [key: string]: any;
}

export interface MatchResult {
  finalScore: number;
  breakdown: {
    semanticScore: number;
    lexicalScore: number;
  };
  matchedKeywords: string[];
}

export interface ChunkingOptions {
  targetChunkSize?: number;
  overlapSize?: number;
}
