import { GoogleGenerativeAI } from '@google/generative-ai';
import { EmbeddingService } from './EmbeddingService';
import { PineconeService } from './PineconeService';
import { db } from './DatabaseService';

export interface ChatResponse {
  answer: string;
}

export class ChatService {
  private genAI: GoogleGenerativeAI;
  private embeddingService: EmbeddingService;
  private pineconeService: PineconeService;
  private maxHistoryMessages: number = 10;
  private topK: number = 10;

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
    const comparison = await db.getComparison(comparisonId);
    if (!comparison) {
      throw new Error(`Comparison ${comparisonId} not found`);
    }

    const session = await db.getOrCreateChatSession(comparisonId);
    const recentMessages = session.messages.slice(-this.maxHistoryMessages);
    
    const questionEmbedding = await this.embeddingService.embedText(userQuestion);

    const resumeChunks = await this.retrieveChunks(
      questionEmbedding,
      comparison.resumeDocId,
      'resume',
      this.topK
    );

    const ragPrompt = this.buildRAGPrompt(
      userQuestion,
      resumeChunks,
      recentMessages,
      comparison
    );

    const answer = await this.callLLM(ragPrompt);

    await db.addChatMessage(comparisonId, 'user', userQuestion);
    await db.addChatMessage(comparisonId, 'assistant', answer);

    return {
      answer,
    };
  }

  private async retrieveChunks(
    queryEmbedding: number[],
    docId: string,
    docType: 'resume' | 'job_description',
    topK: number
  ): Promise<Array<{ chunkId: string; snippet: string; score: number; docType: 'resume' | 'job_description' }>> {
    try {
      const queryFilter = {
        doc_id: docId,
        doc_type: docType,
        chunk_index: { $gte: 0 }
      };

      const matches = await this.pineconeService.query(queryEmbedding, topK, queryFilter);

      const indexMetric = await this.pineconeService.getIndexMetric();

      const normalizedChunks = matches.map(match => {
        const normalizedScore = this.normalizeScore(match.score, indexMetric);
        return {
          chunkId: match.id,
          snippet: match.metadata?.text_snippet || '',
          score: normalizedScore,
          docType: docType
        };
      });

      return normalizedChunks;
    } catch (error) {
      return [];
    }
  }

  private buildRAGPrompt(
    userQuestion: string,
    chunks: Array<{ chunkId: string; snippet: string; score: number; docType: 'resume' | 'job_description' }>,
    recentMessages: Array<{ role: 'user' | 'assistant'; content: string; timestamp: Date }>,
    _comparison: any
  ): string {
    const contextSections = chunks.map((chunk) => {
      return chunk.snippet;
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

Answer:`;

    return prompt;
  }

  private async callLLM(prompt: string): Promise<string> {
    try {
      const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
      const result = await model.generateContent(prompt);
      const response = result.response;
      return response.text();
    } catch (error) {
      throw new Error(`Failed to generate answer: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private normalizeScore(rawScore: number, metric: 'cosine' | 'euclidean' | 'dotproduct'): number {
    if (!Number.isFinite(rawScore)) return 0;
    
    if (metric === 'cosine') {
      return Math.max(0, (rawScore + 1) / 2);
    } else if (metric === 'dotproduct') {
      return Math.max(0, Math.min(1, rawScore));
    } else {
      return Math.max(0, Math.min(1, 1 - rawScore));
    }
  }
}

