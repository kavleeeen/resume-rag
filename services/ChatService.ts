import { GoogleGenerativeAI } from '@google/generative-ai';
import { EmbeddingService } from './EmbeddingService';
import { PineconeService } from './PineconeService';
import { db } from './DatabaseService';
import { normalizeScore } from '../utils/scoreNormalization';

export interface ChatResponse {
  answer: string;
}

export class ChatService {
  private genAI: GoogleGenerativeAI;
  private embeddingService: EmbeddingService;
  private pineconeService: PineconeService;
  private maxHistoryMessages: number = 10;
  private questionTopK: number = 12; // Default from LLD
  private dedupeThreshold: number = 0.95; // For text deduplication
  private mmrLambda: number = 0.6; // MMR diversity parameter

  constructor(
    geminiApiKey: string,
    embeddingService: EmbeddingService,
    pineconeService: PineconeService
  ) {
    this.genAI = new GoogleGenerativeAI(geminiApiKey);
    this.embeddingService = embeddingService;
    this.pineconeService = pineconeService;
  }

  async chat(comparisonId: string, userQuestion: string): Promise<ChatResponse> {
    console.log(`[ChatService] Starting chat for comparison: ${comparisonId}`);

    const comparison = await db.getComparison(comparisonId);
    if (!comparison) {
      console.error(`[ChatService] ERROR: Comparison ${comparisonId} not found`);
      throw new Error(`Comparison ${comparisonId} not found`);
    }

    const session = await db.getOrCreateChatSession(comparisonId);
    const recentMessages = session.messages.slice(-this.maxHistoryMessages);
    
    const questionEmbedding = await this.embeddingService.embedText(userQuestion);

    // Retrieve more chunks initially for MMR/dedup (2x for selection)
    const retrieveTopK = this.questionTopK * 2;

    // Use the same approach as MatchingService: get the actual resume document from DB
    // This ensures we use the correct docId (which might be different if document was reused)
    const resumeDoc = await db.getDocument(comparison.resumeDocId);
    if (!resumeDoc) {
      console.error(`[ChatService] ERROR: Resume document ${comparison.resumeDocId} not found in database`);
      throw new Error(`Resume document ${comparison.resumeDocId} not found in database`);
    }

    // Use the docId from the document record (same as MatchingService does)
    // This is the actual docId that was used for indexing
    const resumeDocId = resumeDoc.docId;

    const resumeChunks = await this.retrieveChunks(
      questionEmbedding,
      resumeDocId,
      'resume',
      retrieveTopK
    );

    // Deduplicate and apply MMR
    const uniqueChunks = this.deduplicateChunks(resumeChunks, this.dedupeThreshold);
    const selectedChunks = this.mmrSelect(uniqueChunks, this.questionTopK, questionEmbedding, this.mmrLambda);

    const ragPrompt = this.buildRAGPrompt(
      userQuestion,
      selectedChunks,
      recentMessages
    );

    const answer = await this.callLLM(ragPrompt);

    await db.addChatMessage(comparisonId, 'user', userQuestion);
    await db.addChatMessage(comparisonId, 'assistant', answer);
    console.log(`[ChatService] Chat completed successfully`);

    return {
      answer,
    };
  }

  private async retrieveChunks(
    queryEmbedding: number[],
    docId: string,
    docType: 'resume' | 'job_description',
    topK: number
  ): Promise<Array<{ chunkId: string; snippet: string; score: number }>> {
    try {
      // Use the same filter format as MatchingService
      const queryFilter = {
        doc_id: docId,
        doc_type: docType,
        chunk_index: { $gte: 0 }
      };

      const matches = await this.pineconeService.query(queryEmbedding, topK, queryFilter, false, true);

      // If no matches with filter, try with only doc_type filter (for debugging)
      if (matches.length === 0) {
        console.warn(`[ChatService] No matches with full filter, trying query with only doc_type='resume'...`);
        const resumeOnlyFilter = {
          doc_type: 'resume',
          chunk_index: { $gte: 0 }
        };
        const resumeOnlyMatches = await this.pineconeService.query(queryEmbedding, Math.min(topK * 2, 100), resumeOnlyFilter, false, true);

        if (resumeOnlyMatches.length === 0) {
          console.warn(`[ChatService] WARNING: No resume chunks found in Pinecone!`);
        }
      }

      const indexMetric = await this.pineconeService.getIndexMetric();

      const normalizedChunks = matches.map(match => {
        const normalizedScore = normalizeScore(match.score, indexMetric);
        return {
          chunkId: match.id,
          snippet: match.metadata?.text_snippet || '',
          score: normalizedScore
        };
      });
      return normalizedChunks;
    } catch (error) {
      console.error(`[ChatService] ERROR: Failed to retrieve chunks:`, error);
      return [];
    }
  }

  private buildRAGPrompt(
    userQuestion: string,
    chunks: Array<{ chunkId: string; snippet: string; score: number }>,
    recentMessages: Array<{ role: 'user' | 'assistant'; content: string; timestamp: Date }>
  ): string {
    const contextSections = chunks.map((chunk) => {
      return `[${chunk.chunkId}] ${chunk.snippet}`;
    }).join('\n\n---\n\n');

    const conversationHistory = recentMessages.length > 0
      ? '\n\nPrevious conversation:\n' +
        recentMessages.map(msg => `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`).join('\n')
      : '';

    const prompt = `You are a helpful assistant answering questions about a resume

Context from resume document:

${contextSections}

Conversation history:
${conversationHistory}

Current Question: ${userQuestion}

Instructions:
- Answer the question based on the provided context from the resume.
- Do NOT include any citations, references, or numbers in brackets like [1], [2], [4, 5], etc. in your answer.
- Answer naturally without any citation markers or references.
- If the information is not in the context, say so clearly.
- Be concise and accurate.
- Use temperature 0.0-0.2 for consistent responses.

Answer:`;

    return prompt;
  }

  private async callLLM(prompt: string): Promise<string> {
    try {
      const model = this.genAI.getGenerativeModel({
        model: 'gemini-2.5-flash',
        generationConfig: {
          temperature: 0.2, // Low temperature for consistent responses
        }
      });
      const result = await model.generateContent(prompt);
      const response = result.response;
      const text = response.text();
      return text;
    } catch (error) {
      console.error(`[ChatService] ERROR: LLM call failed:`, error);
      throw new Error(`Failed to generate answer: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }


  private deduplicateChunks(
    chunks: Array<{ chunkId: string; snippet: string; score: number }>,
    _threshold: number = 0.95
  ): Array<{ chunkId: string; snippet: string; score: number }> {
    const seen = new Set<string>();
    const unique: Array<{ chunkId: string; snippet: string; score: number }> = [];

    for (const chunk of chunks) {
      const normalized = chunk.snippet
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim();

      // Simple hash-based deduplication
      const hash = this.simpleHash(normalized);

      if (!seen.has(hash)) {
        seen.add(hash);
        unique.push(chunk);
      }
    }

    return unique;
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0;i < str.length;i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash.toString();
  }

  private mmrSelect(
    chunks: Array<{ chunkId: string; snippet: string; score: number }>,
    topK: number,
    _queryVector: number[],
    lambda: number = 0.6
  ): Array<{ chunkId: string; snippet: string; score: number }> {
    if (chunks.length <= topK) {
      return chunks;
    }

    // MMR: balance relevance and diversity
    // Score = λ * relevance - (1-λ) * max_similarity_to_selected
    const selected: Array<{ chunkId: string; snippet: string; score: number }> = [];
    const remaining = [...chunks];

    // Start with highest relevance chunk
    remaining.sort((a, b) => b.score - a.score);
    if (remaining.length > 0) {
      selected.push(remaining.shift()!);
    }

    // Greedily select chunks that maximize MMR score
    while (selected.length < topK && remaining.length > 0) {
      let bestIdx = 0;
      let bestScore = -Infinity;

      for (let i = 0;i < remaining.length;i++) {
        const chunk = remaining[i];

        // Find max similarity to already selected chunks
        let maxSimilarity = 0;
        for (const selectedChunk of selected) {
          // Simple text-based similarity (normalized overlap)
          const similarity = this.textSimilarity(chunk.snippet, selectedChunk.snippet);
          maxSimilarity = Math.max(maxSimilarity, similarity);
        }

        // MMR score: λ * relevance - (1-λ) * max_similarity
        const mmrScore = lambda * chunk.score - (1 - lambda) * maxSimilarity;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIdx = i;
        }
      }

      selected.push(remaining.splice(bestIdx, 1)[0]);
    }

    return selected;
  }

  private textSimilarity(text1: string, text2: string): number {
    // Simple Jaccard similarity on words
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }
}

