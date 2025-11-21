import { MongoClient, Db, Collection } from 'mongodb';
import dotenv from 'dotenv';

dotenv.config();

export interface DocumentRecord {
  docId: string;
  type: 'resume' | 'job_description';
  filename: string;
  rawText: string;
  status: 'pending' | 'processing' | 'indexed' | 'failed';
  indexedAt?: Date;
  vectorCount?: number;
  extractedLength?: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface JDRecord {
  docId: string;
  topSkills: string[]; // 4-5 most critical skills based on market standards
  generalSkills: string[]; // Remaining skills
  requiredYears?: number;
  skillSynonyms?: { [skill: string]: string[] }; // LLM-generated synonyms for each skill
  skillEmbeddings?: { [skill: string]: number[] };
  createdAt: Date;
  updatedAt: Date;
}

export interface FileRecord {
  docId: string;
  type: 'resume' | 'job_description';
  originalName: string;
  gcsFilename: string;
  gcsUrl: string;
  gcsLongTermUrl: string;
  size: number;
  mimetype: string;
  expiresAt: string;
  longTermExpiresAt: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface ComparisonRecord {
  comparisonId: string;
  resumeDocId: string;
  jdDocId: string;
  resumeGcsPath?: string; // GCS filename/path for resume (optional if GCS upload fails)
  jdGcsPath?: string; // GCS filename/path for job description (optional if GCS upload fails)
  resumeGcsUrl?: string;
  resumeGcsLongTermUrl?: string;
  jdGcsUrl?: string;
  jdGcsLongTermUrl?: string;
  status: 'in_progress' | 'completed' | 'failed';
  error?: string; // Error message if status is 'failed'
  matchResult?: {
    finalPercent: number;
    semanticScore: number;
    keywordScore: number;
    yearsScore: number;
    resumeYears: number | null;
    jdRequiredYears: number | null;
    matchedSkills: string[];
    missingSkills: string[];
  };
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface ChatSession {
  sessionId: string;
  comparisonId: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

class DatabaseService {
  private client: MongoClient | null = null;
  private db: Db | null = null;
  private documentsCollection: Collection<DocumentRecord> | null = null;
  private jdsCollection: Collection<JDRecord> | null = null;
  private filesCollection: Collection<FileRecord> | null = null;
  private comparisonsCollection: Collection<ComparisonRecord> | null = null;
  private chatSessionsCollection: Collection<ChatSession> | null = null;
  private isConnected = false;

  async connect(): Promise<void> {
    if (this.isConnected && this.db) {
      return;
    }

    const uri = process.env.MONGODB_URI;
    if (!uri) {
      throw new Error('MONGODB_URI environment variable is not set');
    }

    try {
      this.client = new MongoClient(uri);
      await this.client.connect();
      this.db = this.client.db('resume');
      this.documentsCollection = this.db.collection<DocumentRecord>('documents');
      this.jdsCollection = this.db.collection<JDRecord>('jds');
      this.filesCollection = this.db.collection<FileRecord>('files');
      this.comparisonsCollection = this.db.collection<ComparisonRecord>('comparisons');
      this.chatSessionsCollection = this.db.collection<ChatSession>('chat_sessions');
      this.isConnected = true;
    } catch (error) {
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.isConnected = false;
      this.db = null;
      this.documentsCollection = null;
      this.jdsCollection = null;
      this.filesCollection = null;
      this.comparisonsCollection = null;
      this.chatSessionsCollection = null;
    }
  }

  private async ensureConnected(): Promise<void> {
    if (!this.isConnected) {
      await this.connect();
    }
  }

  async createDocument(docId: string, type: 'resume' | 'job_description', filename: string): Promise<DocumentRecord> {
    await this.ensureConnected();
    if (!this.documentsCollection) throw new Error('Database not initialized');

    const doc: DocumentRecord = {
      docId,
      type,
      filename,
      rawText: '',
      status: 'pending',
      createdAt: new Date(),
      updatedAt: new Date()
    };
    await this.documentsCollection.insertOne(doc);
    return doc;
  }

  async getDocument(docId: string): Promise<DocumentRecord | null> {
    await this.ensureConnected();
    if (!this.documentsCollection) throw new Error('Database not initialized');
    return await this.documentsCollection.findOne({ docId });
  }

  async updateDocument(docId: string, updates: Partial<DocumentRecord>): Promise<DocumentRecord> {
    await this.ensureConnected();
    if (!this.documentsCollection) throw new Error('Database not initialized');

    const result = await this.documentsCollection.findOneAndUpdate(
      { docId },
      { $set: { ...updates, updatedAt: new Date() } },
      { returnDocument: 'after' }
    );

    if (!result) {
      throw new Error(`Document ${docId} not found`);
    }
    return result;
  }

  async setRawText(docId: string, rawText: string, extractedLength: number): Promise<void> {
    await this.updateDocument(docId, { rawText, extractedLength });
  }

  async setIndexed(docId: string, vectorCount: number): Promise<void> {
    await this.updateDocument(docId, {
      status: 'indexed',
      indexedAt: new Date(),
      vectorCount
    });
  }

  async setStatus(docId: string, status: DocumentRecord['status']): Promise<void> {
    await this.updateDocument(docId, { status });
  }

  async createJD(docId: string, topSkills: string[], generalSkills: string[], requiredYears?: number): Promise<JDRecord> {
    await this.ensureConnected();
    if (!this.jdsCollection) throw new Error('Database not initialized');

    const jd: JDRecord = {
      docId,
      topSkills,
      generalSkills,
      requiredYears,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    await this.jdsCollection.insertOne(jd);
    return jd;
  }

  async getJD(docId: string): Promise<JDRecord | null> {
    await this.ensureConnected();
    if (!this.jdsCollection) throw new Error('Database not initialized');
    return await this.jdsCollection.findOne({ docId });
  }

  async updateJD(docId: string, updates: Partial<JDRecord>): Promise<JDRecord> {
    await this.ensureConnected();
    if (!this.jdsCollection) throw new Error('Database not initialized');

    const result = await this.jdsCollection.findOneAndUpdate(
      { docId },
      { $set: { ...updates, updatedAt: new Date() } },
      { returnDocument: 'after' }
    );

    if (!result) {
      throw new Error(`JD ${docId} not found`);
    }
    return result;
  }

  async createFileRecord(
    docId: string,
    type: 'resume' | 'job_description',
    originalName: string,
    gcsFilename: string,
    gcsUrl: string,
    gcsLongTermUrl: string,
    size: number,
    mimetype: string,
    expiresAt: string,
    longTermExpiresAt: string
  ): Promise<FileRecord> {
    await this.ensureConnected();
    if (!this.filesCollection) throw new Error('Database not initialized');

    const fileRecord: FileRecord = {
      docId,
      type,
      originalName,
      gcsFilename,
      gcsUrl,
      gcsLongTermUrl,
      size,
      mimetype,
      expiresAt,
      longTermExpiresAt,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    await this.filesCollection.insertOne(fileRecord);
    return fileRecord;
  }

  async getFileRecord(docId: string): Promise<FileRecord | null> {
    await this.ensureConnected();
    if (!this.filesCollection) throw new Error('Database not initialized');
    return await this.filesCollection.findOne({ docId });
  }

  async createComparison(
    comparisonId: string,
    resumeDocId: string,
    jdDocId: string,
    resumeGcsPath?: string,
    jdGcsPath?: string,
    resumeGcsUrl?: string,
    resumeGcsLongTermUrl?: string,
    jdGcsUrl?: string,
    jdGcsLongTermUrl?: string
  ): Promise<ComparisonRecord> {
    await this.ensureConnected();
    if (!this.comparisonsCollection) throw new Error('Database not initialized');

    const comparison: ComparisonRecord = {
      comparisonId,
      resumeDocId,
      jdDocId,
      resumeGcsPath,
      jdGcsPath,
      resumeGcsUrl,
      resumeGcsLongTermUrl,
      jdGcsUrl,
      jdGcsLongTermUrl,
      status: 'in_progress',
      createdAt: new Date(),
      updatedAt: new Date()
    };

    await this.comparisonsCollection.insertOne(comparison);
    return comparison;
  }

  async updateComparisonWithMatch(
    comparisonId: string,
    matchResult: ComparisonRecord['matchResult']
  ): Promise<ComparisonRecord> {
    await this.ensureConnected();
    if (!this.comparisonsCollection) throw new Error('Database not initialized');

    const result = await this.comparisonsCollection.findOneAndUpdate(
      { comparisonId },
      { 
        $set: { 
          matchResult,
          status: 'completed',
          updatedAt: new Date() 
        } 
      },
      { returnDocument: 'after' }
    );

    if (!result) {
      throw new Error(`Comparison ${comparisonId} not found`);
    }
    return result;
  }

  async updateComparisonStatus(
    comparisonId: string,
    status: 'in_progress' | 'completed' | 'failed',
    error?: string
  ): Promise<ComparisonRecord> {
    await this.ensureConnected();
    if (!this.comparisonsCollection) throw new Error('Database not initialized');

    const update: any = {
      status,
      updatedAt: new Date()
    };

    if (error) {
      update.error = error;
    }

    const result = await this.comparisonsCollection.findOneAndUpdate(
      { comparisonId },
      { $set: update },
      { returnDocument: 'after' }
    );

    if (!result) {
      throw new Error(`Comparison ${comparisonId} not found`);
    }
    return result;
  }

  async getComparison(comparisonId: string): Promise<ComparisonRecord | null> {
    await this.ensureConnected();
    if (!this.comparisonsCollection) throw new Error('Database not initialized');
    return await this.comparisonsCollection.findOne({ comparisonId });
  }

  async getComparisonByDocIds(resumeDocId: string, jdDocId: string): Promise<ComparisonRecord | null> {
    await this.ensureConnected();
    if (!this.comparisonsCollection) throw new Error('Database not initialized');
    return await this.comparisonsCollection.findOne({ resumeDocId, jdDocId });
  }

  async getOrCreateChatSession(comparisonId: string): Promise<ChatSession> {
    await this.ensureConnected();
    if (!this.chatSessionsCollection) throw new Error('Database not initialized');

    let session = await this.chatSessionsCollection.findOne({ comparisonId });

    if (!session) {
      const sessionId = `chat:${comparisonId}`;
      session = {
        sessionId,
        comparisonId,
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date()
      };
      await this.chatSessionsCollection.insertOne(session);
    }

    return session;
  }

  async addChatMessage(comparisonId: string, role: 'user' | 'assistant', content: string): Promise<ChatSession> {
    await this.ensureConnected();
    if (!this.chatSessionsCollection) throw new Error('Database not initialized');

    const message: ChatMessage = {
      role,
      content,
      timestamp: new Date()
    };

    const result = await this.chatSessionsCollection.findOneAndUpdate(
      { comparisonId },
      {
        $push: { messages: message },
        $set: { updatedAt: new Date() }
      },
      { returnDocument: 'after', upsert: true }
    );

    if (!result) {
      throw new Error(`Failed to add chat message for comparison ${comparisonId}`);
    }

    return result;
  }

  async getChatSession(comparisonId: string): Promise<ChatSession | null> {
    await this.ensureConnected();
    if (!this.chatSessionsCollection) throw new Error('Database not initialized');
    return await this.chatSessionsCollection.findOne({ comparisonId });
  }
}

export const db = new DatabaseService();

