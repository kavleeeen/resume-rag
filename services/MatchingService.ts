import { PineconeService } from './PineconeService';
import { db } from './DatabaseService';
import { EmbeddingService } from './EmbeddingService';
import { normalizeScore as normalizeScoreUtil } from '../utils/scoreNormalization';

export interface MatchResult {
  resumeId: string;
  jdId: string;
  finalPercent: number;
  semanticScore: number;
  keywordScore: number;
  yearsScore: number;
  resumeYears?: number | null;
  jdRequiredYears?: number | null;
  matchedSkills: string[];
  missingSkills: string[];
  topChunks: Array<{
    chunkId: string;
    snippet: string;
    score: number;
  }>;
  explain: string;
}

export class MatchingService {
  private pineconeService: PineconeService;
  private embeddingService: EmbeddingService;
  private indexMetric: 'cosine' | 'euclidean' | 'dotproduct' | null = null;

  constructor(
    pineconeService: PineconeService,
    embeddingService: EmbeddingService
  ) {
    this.pineconeService = pineconeService;
    this.embeddingService = embeddingService;
    // Cache index metric in constructor (or init)
    this.pineconeService.getIndexMetric()
      .then(metric => {
        this.indexMetric = metric;
        console.log('[MatchingService] Detected metric:', metric);
      })
      .catch(err => {
        console.warn('[MatchingService] Failed to get metric, defaulting to cosine', err);
        this.indexMetric = 'cosine';
      });
  }

  /**
   * Get the cached index metric, initializing if needed
   */
  private async getIndexMetric(): Promise<'cosine' | 'euclidean' | 'dotproduct'> {
    if (this.indexMetric !== null) {
      return this.indexMetric;
    }
    // Wait a bit for initialization if in progress, then return default
    await new Promise(resolve => setTimeout(resolve, 100));
    if (this.indexMetric !== null) {
      return this.indexMetric;
    }
    // Ultimate fallback
    return 'cosine';
  }

  /**
   * Calculate match score between a Resume and Job Description
   * 
   * How Semantic Matching Works (Step-by-Step):
   * 
   * 1️⃣ Break documents into chunks
   *    - Resume: Split into chunks (40-150 vectors) - each work section, skill block, project, etc.
   *    - JD: Stored as 1 doc-level embedding (summarized meaning representation)
   * 
   * 2️⃣ Convert text → embeddings
   *    - Use language model (Gemini) to convert text into vectors [0.12, -0.33, 0.88, ...]
   *    - Vectors with similar meanings lie closer together in space
   * 
   * 3️⃣ Compare embeddings using similarity search
   *    - Query Pinecone with JD vector
   *    - Find closest resume chunks
   *    - If closest chunks match required skills/responsibilities → Strong semantic match
   *    - Similarity metric (cosine): Same meaning → 1.0, Different → 0.0
   * 
   * 4️⃣ Aggregation of top semantic matches
   *    - Take top relevant resume chunks
   *    - Normalize scores to [0-1]
   *    - Use meanTopN(3) to get final semantic score
   * 
   * Final Score = weighted combination of:
   * - Semantic Score: Conceptual match (catches "AWS" vs "EC2 via Kubernetes")
   * - Keyword Score: Exact skill matches
   * - Years Score: Experience requirement match
   */
  async calculateMatch(
    resumeId: string,
    jdId: string,
    weights: { semantic: number; keyword: number; years: number } = {
      semantic: 0.4,
      keyword: 0.55,
      years: 0.05
    }
  ): Promise<MatchResult> {
    console.log(`[MatchingService] Starting match calculation - Resume: ${resumeId}, JD: ${jdId}`);

    // Verify documents exist and are indexed
    const resumeDoc = await db.getDocument(resumeId);
    const jdDoc = await db.getDocument(jdId);

    if (!resumeDoc) {
      console.error(`[MatchingService] ERROR: Resume document ${resumeId} not found in database`);
      throw new Error(`Resume document ${resumeId} not found in database`);
    }
    if (!jdDoc) {
      console.error(`[MatchingService] ERROR: JD document ${jdId} not found in database`);
      throw new Error(`JD document ${jdId} not found in database`);
    }

    if (resumeDoc.status !== 'indexed') {
      console.error(`[MatchingService] ERROR: Resume ${resumeId} is not indexed yet. Status: ${resumeDoc.status}`);
      throw new Error(`Resume ${resumeId} is not indexed yet. Status: ${resumeDoc.status}`);
    }
    if (jdDoc.status !== 'indexed') {
      console.error(`[MatchingService] ERROR: JD ${jdId} is not indexed yet. Status: ${jdDoc.status}`);
      throw new Error(`JD ${jdId} is not indexed yet. Status: ${jdDoc.status}`);
    }

    // Get cached index metric (initialized once on construction)
    const indexMetric = await this.getIndexMetric();

    const jdVector = await this.getJDVector(jdId, jdDoc);

    // Verify embedding dimension matches expected
    const expectedDimension = this.embeddingService.getDimension();
    if (jdVector.length !== expectedDimension) {
      console.error(`[MatchingService] ERROR: JD vector dimension mismatch: expected ${expectedDimension}, got ${jdVector.length}`);
      throw new Error(`JD vector dimension mismatch: expected ${expectedDimension}, got ${jdVector.length}. Possible embedding model mismatch.`);
    }

    /**
     * Step 3: Compare embeddings using similarity search
     * 
     * Process:
     * - Provide JD vector (doc-level embedding representing entire job description)
     * - Pinecone finds closest resume chunks (chunk-level embeddings)
     * - If closest chunks are about required skills/responsibilities → Strong semantic match
     * 
     * Similarity metric (cosine):
     * - Same meaning → score ≈ 1.0
     * - Different meaning → score ≈ 0.0
     * - Opposite → negative scores (normalized to [0,1])
     * 
     * Query top ~20 chunks for semantic matching (we'll use more for keyword matching)
     */
    const chunkTopK = 20; // Top ~20 relevant chunks for semantic score aggregation
    const queryFilter = {
        doc_id: resumeId,
        doc_type: 'resume',
        chunk_index: { $gte: 0 }
      };

    let allMatches: Array<{ id: string; score: number; metadata: any }> = [];

      try {
        allMatches = await this.pineconeService.query(jdVector, chunkTopK, queryFilter, false, true);
      } catch (error) {
        console.warn(`[MatchingService] Query failed, trying fallback method:`, error);
        try {
          const resumeDoc = await db.getDocument(resumeId);
          if (!resumeDoc) {
            throw new Error(`Resume document ${resumeId} not found for fallback fetch`);
          }
          const expectedChunkCount = (resumeDoc.vectorCount || 200) - 1; // Subtract 1 for docvec
          const maxChunksToFetch = Math.min(expectedChunkCount, 200);
          // Use canonical docId from DB record (_id or docId field)
          const canonicalDocId = (resumeDoc as any)._id || resumeDoc.docId || resumeId;
          const chunkIds = Array.from({ length: maxChunksToFetch }, (_, i) => `${canonicalDocId}::chunk::${i}`);

          const fetchedChunks = await this.pineconeService.fetchByIds(chunkIds);

          // Validate vector dims on fallback cosine computation
          const expectedDimension = this.embeddingService.getDimension();
          allMatches = fetchedChunks
            .filter(chunk => {
              if (jdVector.length !== chunk.values.length) {
                console.warn(`[MatchingService] Dim mismatch jdVec vs chunkVec: ${jdVector.length} vs ${chunk.values.length}, jdId: ${jdId}, chunkId: ${chunk.id}`);
                return false; // Skip this chunk
              }
              if (chunk.values.length !== expectedDimension) {
                console.warn(`[MatchingService] Skipping chunk ${chunk.id}: dimension mismatch (expected ${expectedDimension}, got ${chunk.values.length})`);
                return false;
              }
              return true;
            })
            .map(chunk => {
              const cosineSimilarity = this.computeCosineSimilarity(jdVector, chunk.values);
              // Use same normalization as normalizeScoreUtil for cosine
              const normalizedScore = normalizeScoreUtil(cosineSimilarity, 'cosine');
              return {
                id: chunk.id,
                score: normalizedScore,
                metadata: chunk.metadata
              };
            })
            .sort((a, b) => b.score - a.score);
        } catch (err) {
          console.error(`[MatchingService] Fallback method also failed:`, err);
          throw err;
      }
    }

    /**
     * Step 4: Aggregation of top semantic matches
     * 
     * Process:
     * 1. Take top ~20 relevant resume chunks
     * 2. Normalize scores to [0-1]
     * 3. Weighted average + quality/coverage boost
     * 
     * Semantic Matching Process:
     * 1. Resume is split into chunks (40-150 vectors) - each work section, skill block, project, etc.
     * 2. JD is stored as 1 doc-level embedding (summarized meaning representation)
     * 3. We query Pinecone with JD vector to find closest resume chunks
     * 4. Similarity scores indicate conceptual match:
     *    - Score ≈ 1.0: Same meaning (very strong match)
     *    - Score ≈ 0.6-0.85: Good but not complete match
     *    - Score ≈ 0.4-0.6: Weak/partial relevance
     *    - Score < 0.4: Conceptually unrelated
     * 
     * Why semantic matching is needed:
     * - Resumes rarely copy JD keywords exactly
     * - Example: JD says "AWS experience" but resume says "EC2 via Kubernetes"
     * - Vector embedding similarity captures the conceptual match even without exact keywords
     */
    const normalizedMatches = allMatches.map(match => {
      const normalized = normalizeScoreUtil(match.score, indexMetric);
      return {
        ...match,
        score: normalized
      };
    });

    // Take top ~20 relevant resume chunks for aggregation
    const topChunksForSemantic = 20;
    const semanticScore = this.computeSemanticScoreWithBoosts(normalizedMatches.slice(0, topChunksForSemantic));

    const topMatches = normalizedMatches.slice(0, 20);
    const { keywordScore, matchedSkills, missingSkills } = await this.computeKeywordScore(
      jdId,
      resumeId,
      topMatches
    );

    const jdRecord = await db.getJD(jdId);
    const jdRequiredYears = jdRecord?.requiredYears ?? null;
    const yearsResult = await this.computeYearsScore(jdId, resumeId);
    const yearsScore = yearsResult.score;
    const resumeYears = yearsResult.resumeYears ?? null;

    const finalScore = Math.max(
      0,
      Math.min(
        1,
        weights.semantic * semanticScore +
          weights.keyword * keywordScore +
          weights.years * yearsScore
      )
    );
    const finalPercent = Math.round(finalScore * 100);

    const explain = this.generateExplanation(
      semanticScore,
      keywordScore,
      yearsScore,
      matchedSkills,
      missingSkills,
      topMatches
    );

    const result = {
      resumeId,
      jdId,
      finalPercent,
      semanticScore,
      keywordScore,
      yearsScore,
      resumeYears,
      jdRequiredYears,
      matchedSkills,
      missingSkills,
      topChunks: topMatches.slice(0, 3).map(match => ({
        chunkId: match.id,
        snippet: match.metadata.text_snippet || '',
        score: match.score
      })),
      explain
    };

    console.log(`[MatchingService] Match calculation completed: ${finalPercent}% match (semantic: ${semanticScore.toFixed(4)}, keyword: ${keywordScore.toFixed(4)}, years: ${yearsScore.toFixed(4)})`);

    return result;
  }

  /**
   * Step 2: Convert text → embeddings
   * 
   * Uses a language model (Gemini) to convert JD text into a vector of numbers.
   * Example: [0.12, -0.33, 0.88, ...] with ~768 or 1024 dimensions
   * 
   * Vectors with similar meanings lie closer together in space.
   * This allows us to find conceptually similar content even without exact keyword matches.
   * 
   * JD is stored as 1 doc-level embedding (summarized into a single meaning representation)
   */
  private async getJDVector(jdId: string, jdDoc: any): Promise<number[]> {
    // Use canonical docId from DB record (_id or docId field)
    const canonicalDocId = (jdDoc as any)._id || jdDoc.docId || jdId;
    const docvecId = `${canonicalDocId}::docvec`;

    try {
      const fetched = await this.pineconeService.fetchById(docvecId);
      if (fetched && fetched.values && Array.isArray(fetched.values) && fetched.values.length > 0) {
        return fetched.values;
      }
    } catch (error) {
      console.warn(`[MatchingService] Failed to fetch JD vector from Pinecone:`, error);
    }
    if (jdDoc && jdDoc.rawText) {
      const vec = await this.embeddingService.embedText(jdDoc.rawText);

      try {
        const uploadedAt = new Date().toISOString();
        const canonicalDocId = (jdDoc as any)._id || jdDoc.docId || jdId;
        await this.pineconeService.upsertChunks([{
          id: docvecId,
          vector: vec,
          metadata: {
            doc_id: canonicalDocId, // Use canonical docId
            doc_type: 'job_description',
            chunk_index: -1,
            version: 'v1',
            uploaded_at: uploadedAt,
            full_text_length: jdDoc.rawText.length
          }
        }]);
      } catch (error) {
        console.warn(`[MatchingService] Failed to store JD vector in Pinecone:`, error);
      }

      return vec;
    }

    console.error(`[MatchingService] ERROR: Cannot get JD vector - no docvec and no raw text available for JD: ${jdId}`);
    throw new Error(`Cannot get JD vector: no docvec and no raw text available for JD: ${jdId}`);
  }

  private computeCosineSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length) {
      throw new Error(`Vector dimension mismatch: ${vec1.length} vs ${vec2.length}`);
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0;i < vec1.length;i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    if (denominator === 0) return 0;

    return dotProduct / denominator;
  }




  private async computeKeywordScore(
    jdId: string,
    resumeId: string,
    _topMatches: Array<{ metadata: any }>
  ): Promise<{ keywordScore: number; matchedSkills: string[]; missingSkills: string[] }> {
    const jdRecord = await db.getJD(jdId);
    if (!jdRecord || (jdRecord.topSkills.length === 0 && jdRecord.generalSkills.length === 0)) {
      return { keywordScore: 0, matchedSkills: [], missingSkills: [] };
    }

    const allSkills = [...jdRecord.topSkills, ...jdRecord.generalSkills];
    const isTopSkill = new Set(jdRecord.topSkills);

    const skillSynonyms = jdRecord.skillSynonyms || {};

    const skillEmbeddings = await this.getOrCreateSkillEmbeddings(jdId, allSkills);
    const semanticMatches = await this.matchSkillsSemantically(resumeId, allSkills, skillEmbeddings);

    const matchedSkills: string[] = [];
    const missingSkills: string[] = [];
    // Tuning defaults: start conservative and tune with data
    const semanticThreshold = 0.6; // Start 0.6
    const evidenceThreshold = 0.6; // Start 0.6

    for (const skill of allSkills) {
      const semanticMatch = semanticMatches[skill];
      const synonyms = skillSynonyms[skill] || [skill];

      let isMatched = false;

      if (semanticMatch && semanticMatch.evidenceChunks.length > 0) {
        const scores = semanticMatch.evidenceChunks.map(chunk => chunk.score);
        const aggScore = this.meanTopN(scores, 3); // meanTopN use N = 3
        const topScore = semanticMatch.evidenceChunks[0].score;

        // Check all evidence chunks for keyword matches (OR condition - additional validation)
        let keywordMatch = false;
        for (const chunk of semanticMatch.evidenceChunks.slice(0, 5)) {
          if (this.evidenceContainsSkillOrSynonym(chunk.snippet, skill, synonyms)) {
            keywordMatch = true;
            break;
          }
        }

        // Match if: (1) Strong semantic match OR (2) Keyword match found
        if (aggScore >= semanticThreshold && topScore >= evidenceThreshold) {
          // Strong semantic match - add to matched skills
          isMatched = true;
        } else if (keywordMatch) {
          // Keyword match found (additional validation) - add to matched skills
          isMatched = true;
        }
      }

      if (isMatched) {
        matchedSkills.push(skill);
      } else {
        missingSkills.push(skill);
      }
    }

    const topSkillsMatched = matchedSkills.filter(skill => isTopSkill.has(skill)).length;
    const generalSkillsMatched = matchedSkills.filter(skill => !isTopSkill.has(skill)).length;

    const totalWeightedMatches = (topSkillsMatched * 2) + generalSkillsMatched;
    const totalWeightedSkills = (jdRecord.topSkills.length * 2) + jdRecord.generalSkills.length;

    const keywordScore = totalWeightedSkills > 0
      ? totalWeightedMatches / totalWeightedSkills
      : 0;

    return { keywordScore, matchedSkills, missingSkills };
  }

  private async getOrCreateSkillEmbeddings(
    jdId: string,
    skills: string[]
  ): Promise<{ [skill: string]: number[] }> {
    const jdRecord = await db.getJD(jdId);

    if (jdRecord?.skillEmbeddings) {
      return jdRecord.skillEmbeddings;
    }

    const embeddings: { [skill: string]: number[] } = {};

    // Generate embeddings in parallel for faster processing
    const skillTexts = skills.map(skill => `Skill: ${skill} - technical expertise and experience`);
    const skillEmbeddingsArray = await this.embeddingService.embedTexts(skillTexts, skills.length);

    // Map embeddings back to skills
    for (let i = 0;i < skills.length;i++) {
      embeddings[skills[i]] = skillEmbeddingsArray[i];
    }
    if (jdRecord) {
      await db.updateJD(jdId, { skillEmbeddings: embeddings });
    }

    return embeddings;
  }

  private async matchSkillsSemantically(
    resumeId: string,
    skills: string[],
    skillEmbeddings: { [skill: string]: number[] }
  ): Promise<{ [skill: string]: { similarity: number; evidenceChunks: Array<{ chunkId: string; snippet: string; score: number }> } }> {
    const results: { [skill: string]: { similarity: number; evidenceChunks: Array<{ chunkId: string; snippet: string; score: number }> } } = {};
    const targetTopK = 10; // Final chunks to use for evaluation
    const retrieveTopK = targetTopK * 2; // Retrieve more for MMR/dedup (same as ChatService)
    const dedupeThreshold = 0.95; // Same as ChatService
    const mmrLambda = 0.6; // Same as ChatService

    for (let i = 0;i < skills.length;i++) {
      const skill = skills[i];
      const skillEmbedding = skillEmbeddings[skill];
      if (!skillEmbedding) {
        console.warn(`[MatchingService] No embedding found for skill: ${skill}`);
        continue;
      }

      try {
        const queryFilter = {
          doc_id: resumeId,
          doc_type: 'resume',
          chunk_index: { $gte: 0 }
        };

        // Retrieve more chunks initially for MMR/dedup
        let matches = await this.pineconeService.query(skillEmbedding, retrieveTopK, queryFilter, false, true);

        if (matches.length === 0) {
          const allMatches = await this.pineconeService.query(skillEmbedding, 200);
          matches = allMatches.filter(m =>
            m.metadata?.doc_id === resumeId &&
            m.metadata?.doc_type === 'resume' &&
            (m.metadata?.chunk_index ?? -1) >= 0
          );
        }

        if (matches.length > 0) {
          const currentIndexMetric = await this.getIndexMetric();

          // Normalize scores
          const normalizedChunks = matches.map(match => ({
            chunkId: match.id,
            snippet: match.metadata?.text_snippet || '',
            score: normalizeScoreUtil(match.score, currentIndexMetric)
          }));

          // Apply deduplication (same as ChatService)
          const uniqueChunks = this.deduplicateChunks(normalizedChunks, dedupeThreshold);

          // Apply MMR selection for diversity (same as ChatService)
          const selectedChunks = this.mmrSelect(uniqueChunks, targetTopK, skillEmbedding, mmrLambda);

          const normalizedScores = selectedChunks.map(chunk => chunk.score);
          const maxSimilarity = Math.max(...normalizedScores);

          results[skill] = {
            similarity: maxSimilarity,
            evidenceChunks: selectedChunks
          };
        } else {
          results[skill] = {
            similarity: 0,
            evidenceChunks: []
          };
        }
      } catch (error) {
        console.error(`[MatchingService] Error matching skill "${skill}":`, error);
        results[skill] = {
          similarity: 0,
          evidenceChunks: []
        };
      }
    }

    return results;
  }

  /**
   * Deduplicate chunks based on text similarity (same logic as ChatService)
   */
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

      const hash = this.simpleHash(normalized);

      if (!seen.has(hash)) {
        seen.add(hash);
        unique.push(chunk);
      }
    }

    return unique;
  }

  /**
   * Simple hash function for deduplication (same as ChatService)
   */
  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0;i < str.length;i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return hash.toString();
  }

  /**
   * MMR (Maximal Marginal Relevance) selection for diversity (same logic as ChatService)
   */
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

  /**
   * Text similarity for MMR (Jaccard similarity on words, same as ChatService)
   */
  private textSimilarity(text1: string, text2: string): number {
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }


  private normalizeSkillName(skill: string): string {
    return skill
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .replace(/[\s\-_]+/g, '')
      .trim();
  }

  private escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  private evidenceContainsSkillOrSynonym(snippet: string, skill: string, synonyms: string[]): boolean {
    const s = snippet.toLowerCase();

    const allTerms = [skill, ...synonyms.filter(syn => syn !== skill)];

    for (const term of allTerms) {
      const termLower = term.toLowerCase();
      const normalizedTerm = this.normalizeSkillName(termLower);
      const normalizedSnippet = this.normalizeSkillName(s);

      if (s.includes(termLower)) {
        return true;
      }

      if (normalizedSnippet.includes(normalizedTerm)) {
        return true;
      }

      const wordBoundaryPattern = new RegExp(`\\b${this.escapeRegex(termLower)}\\b`, 'i');
      if (wordBoundaryPattern.test(s)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Calculate mean of top N scores
   * Used for aggregating semantic match scores from top relevant chunks
   */
  private meanTopN(scores: number[], n: number): number {
    if (scores.length === 0) return 0;
    const topN = scores.slice(0, n);
    return topN.reduce((sum, score) => sum + score, 0) / topN.length;
  }

  /**
   * Compute semantic score with weighted average + quality/coverage boost
   * 
   * Takes top ~20 relevant resume chunks and:
   * 1. Applies weighted average (higher weights for top matches)
   * 2. Adds quality boost for high-scoring matches
   * 3. Adds coverage boost for having multiple relevant chunks
   */
  private computeSemanticScoreWithBoosts(matches: Array<{ score: number }>): number {
    if (matches.length === 0) return 0;

    const relevanceThreshold = 0.2;
    const relevantMatches = matches
      .map(m => Math.max(0, Math.min(1, m.score)))
      .filter(score => score >= relevanceThreshold);

    if (relevantMatches.length === 0) {
      const topScore = Math.max(0, Math.min(1, matches[0]?.score || 0));
      return topScore * 0.5;
  }

    const maxChunksToConsider = Math.min(20, relevantMatches.length);
    const chunksToUse = relevantMatches.slice(0, maxChunksToConsider);

    // Weighted average: higher weights for top matches
    const weights = [0.4, 0.25, 0.15, 0.1, 0.05, 0.025, 0.015, 0.01, 0.005, 0.003];
    let weightedSum = 0;
    let totalWeight = 0;

    for (let i = 0;i < chunksToUse.length;i++) {
      const score = chunksToUse[i];
      const weight = i < weights.length ? weights[i] : Math.exp(-(i - weights.length) * 0.2) * weights[weights.length - 1];

      weightedSum += score * weight;
      totalWeight += weight;
    }

    const mean = totalWeight > 0 ? weightedSum / totalWeight : 0;

    // Quality boost: reward high-quality matches
    // Fix #3: Require at least 3 chunks ≥ 0.70 to give quality boost
    const highQualityMatches = chunksToUse.filter(s => s >= 0.7).length;
    const veryHighQualityMatches = chunksToUse.filter(s => s >= 0.85).length;
    const excellentMatches = chunksToUse.filter(s => s >= 0.9).length;

    let qualityBoost = 0;
    if (highQualityMatches >= 3) {
      // Only give quality boost if we have at least 3 high-quality matches
      qualityBoost = Math.min(0.2,
        highQualityMatches * 0.015 +
        veryHighQualityMatches * 0.025 +
        excellentMatches * 0.03
      );
    }

    // Coverage boost: reward having multiple relevant chunks
    // Fix #2: Reduced from 0.15 to 0.05
    const coverageBoost = Math.min(0.05, (chunksToUse.length / 15) * 0.05);

    // Fix #4: Cap final score BEFORE clamping to prevent false strong matches
    let score = mean + qualityBoost + coverageBoost;
    if (score > 0.9 && mean < 0.7) {
      score = 0.85; // Prevent false strong matches
    }
    return Math.max(0, Math.min(1, score));
    }


  private async extractResumeYears(resumeId: string): Promise<number | undefined> {
    const resumeDoc = await db.getDocument(resumeId);
    if (!resumeDoc || !resumeDoc.rawText) {
      return undefined;
    }

    const resumeText = resumeDoc.rawText.toLowerCase();
    const yearsPatterns = [
      /(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)/gi, // "5 years of experience"
      /(\d+)\+?\s*years?/gi, // "5 years" or "5+ years"
      /experience[:\s]+(\d+)\s*years?/gi, // "Experience: 5 years"
      /(\d+)\s*y\.?o\.?/gi,
    ];

    let maxYears = 0;

    for (const pattern of yearsPatterns) {
      const matches = Array.from(resumeText.matchAll(pattern));
      for (const match of matches) {
        const years = parseInt(match[1] || '0', 10);
        if (years > 0 && years <= 50) {
          maxYears = Math.max(maxYears, years);
        }
      }
    }

    if (maxYears === 0) {
      maxYears = this.extractYearsFromDates(resumeText);
    }

    const result = maxYears > 0 ? maxYears : undefined;
    return result;
  }

  private async computeYearsScore(
    jdId: string,
    resumeId: string
  ): Promise<{ score: number; resumeYears: number | undefined }> {
    const jdRecord = await db.getJD(jdId);
    if (!jdRecord || !jdRecord.requiredYears) {
      const resumeYears = await this.extractResumeYears(resumeId);
      return { score: 1.0, resumeYears };
    }

    const resumeDoc = await db.getDocument(resumeId);
    if (!resumeDoc || !resumeDoc.rawText) {
      console.warn(`[MatchingService] Resume document or raw text not found, returning zero score`);
      return { score: 0, resumeYears: undefined };
    }

    const resumeText = resumeDoc.rawText.toLowerCase();
    const yearsPatterns = [
      /(\d+)\s*years?\s*(?:of\s*)?(?:experience|exp)/gi, // "5 years of experience"
      /(\d+)\+?\s*years?/gi, // "5 years" or "5+ years"
      /experience[:\s]+(\d+)\s*years?/gi, // "Experience: 5 years"
      /(\d+)\s*y\.?o\.?/gi,
    ];

    let maxYears = 0;

    for (const pattern of yearsPatterns) {
      const matches = Array.from(resumeText.matchAll(pattern));
      for (const match of matches) {
        const years = parseInt(match[1] || '0', 10);
        if (years > 0 && years <= 50) {
          maxYears = Math.max(maxYears, years);
        }
      }
    }

    if (maxYears === 0) {
      maxYears = this.extractYearsFromDates(resumeText);
    }

    if (maxYears === 0) {
      return { score: 0, resumeYears: undefined };
    }

    const ratio = maxYears / jdRecord.requiredYears;

    let score: number;
    if (ratio >= 1.0) {
      score = 1.0;
    } else if (ratio >= 0.8) {
      score = 0.9;
    } else if (ratio >= 0.6) {
      score = 0.7;
    } else {
      score = Math.max(0, ratio * 0.8);
    }

    return { score, resumeYears: maxYears };
  }

  private extractYearsFromDates(text: string): number {
    const dateRangePattern = /(\d{4}|\w+\s+\d{4})\s*[-–—]\s*(\d{4}|\w+\s+\d{4}|present|current)/gi;
    const rangeMatches = Array.from(text.matchAll(dateRangePattern));

    if (rangeMatches.length > 0) {
      let totalMonths = 0;
      const currentYear = new Date().getFullYear();
      const currentMonth = new Date().getMonth();

      for (let i = 0;i < rangeMatches.length;i++) {
        const match = rangeMatches[i];
        const startStr = match[1];
        const endStr = match[2].toLowerCase();

        const startYearMatch = startStr.match(/\d{4}/);
        if (!startYearMatch) continue;
        const startYear = parseInt(startYearMatch[0], 10);
        const startMonth = this.extractMonth(startStr) || 0;

        let endYear = currentYear;
        let endMonth = currentMonth;

        if (endStr === 'present' || endStr === 'current') {
          endYear = currentYear;
          endMonth = currentMonth;
        } else {
          const endYearMatch = endStr.match(/\d{4}/);
          if (endYearMatch) {
            endYear = parseInt(endYearMatch[0], 10);
            endMonth = this.extractMonth(endStr) || 11;
          }
        }

        const months = (endYear - startYear) * 12 + (endMonth - startMonth);
        totalMonths += Math.max(0, months);
      }

      const years = Math.ceil(totalMonths / 12);
      return years;
    }

    const monthNames = [
      'january', 'february', 'march', 'april', 'may', 'june',
      'july', 'august', 'september', 'october', 'november', 'december'
    ];
    const monthAbbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

    const monthYearPattern = new RegExp(
      `\\b(${monthNames.join('|')}|${monthAbbr.join('|')})\\.?\\s+(\\d{4})\\b`,
      'gi'
    );

    const allMatches = Array.from(text.matchAll(monthYearPattern));

    if (allMatches.length === 0) {
      return 0;
    }

    const dates: Array<{ year: number; month: number }> = [];
    const currentYear = new Date().getFullYear();
    const currentMonth = new Date().getMonth();

    for (const match of allMatches) {
      const monthStr = match[1].toLowerCase();
      const year = parseInt(match[2], 10);

      let monthIndex = -1;
      for (let i = 0;i < monthNames.length;i++) {
        if (monthStr.includes(monthNames[i]) || monthStr.includes(monthAbbr[i])) {
          monthIndex = i;
          break;
        }
      }

      if (monthIndex >= 0 && year >= 1900 && year <= currentYear + 1) {
        dates.push({ year, month: monthIndex });
      }
    }

    if (dates.length === 0) {
      return 0;
    }

    dates.sort((a, b) => {
      if (a.year !== b.year) return a.year - b.year;
      return a.month - b.month;
    });

    const earliest = dates[0];

    const endYear = currentYear;
    const endMonth = currentMonth;

    const totalMonths = (endYear - earliest.year) * 12 + (endMonth - earliest.month);
    const years = Math.ceil(totalMonths / 12);

    return years;
  }

  private extractMonth(str: string): number | null {
    const monthNames = [
      'january', 'february', 'march', 'april', 'may', 'june',
      'july', 'august', 'september', 'october', 'november', 'december'
    ];
    const monthAbbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

    const lower = str.toLowerCase();
    for (let i = 0;i < monthNames.length;i++) {
      if (lower.includes(monthNames[i]) || lower.includes(monthAbbr[i])) {
        return i;
      }
    }
    return null;
  }

  private generateExplanation(
    semanticScore: number,
    keywordScore: number,
    yearsScore: number,
    matchedSkills: string[],
    missingSkills: string[],
    topMatches: Array<{ id: string; score: number; metadata: any }>
  ): string {
    const parts: string[] = [];

    if (semanticScore >= 0.7) {
      parts.push('Strong semantic match');
    } else if (semanticScore >= 0.5) {
      parts.push('Moderate semantic match');
    } else {
      parts.push('Weak semantic match');
    }

    if (keywordScore >= 0.8) {
      parts.push('excellent skill match');
    } else if (keywordScore >= 0.6) {
      parts.push('good skill match');
    } else if (keywordScore >= 0.4) {
      parts.push('partial skill match');
    }

    if (matchedSkills.length > 0) {
      parts.push(`matched ${matchedSkills.length} required skills: ${matchedSkills.slice(0, 3).join(', ')}`);
    } else {
      parts.push('no required skills matched');
    }

    if (missingSkills.length > 0) {
      parts.push(`missing ${missingSkills.length} required skills: ${missingSkills.slice(0, 3).join(', ')}`);
    }

    if (yearsScore >= 1.0) {
      parts.push('years requirement fully met');
    } else if (yearsScore >= 0.8) {
      parts.push('years requirement nearly met');
    } else if (yearsScore > 0) {
      parts.push('years requirement partially met');
    } else {
      parts.push('years requirement not met');
    }

    // Add info about match quality
    if (topMatches.length > 0 && topMatches[0].score >= 0.8) {
      parts.push('high-quality content matches found');
    }

    return parts.join('; ') + '.';
  }
}
