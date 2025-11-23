export interface ChunkMetadata {
  // Document identification
  doc_id: string; // e.g., "resume:1234" or "jd:5678"
  doc_type: 'resume' | 'job_description' | 'document';
  chunk_index: number;
  
  section?: string;
  text_snippet?: string; // First 300 chars
  version?: string; // e.g., "v1"
  uploaded_at?: string; // ISO timestamp
  
  source?: string;
  filename?: string;
  mimetype?: string;
  pages?: number;
  estimated_tokens?: number;
  has_overlap?: boolean;
  overlap_size?: number;
  full_text_length?: number;
  embedding_model?: string;
  embedding_dimension?: number;
}
