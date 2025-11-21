import { PineconeService } from './PineconeService';
import { db } from './DatabaseService';
import { EmbeddingService } from './EmbeddingService';

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
  private indexMetric: 'cosine' | 'euclidean' | 'dotproduct' = 'cosine';

  constructor(
    pineconeService: PineconeService,
    embeddingService: EmbeddingService
  ) {
    this.pineconeService = pineconeService;
    this.embeddingService = embeddingService;
  }

  async calculateMatch(
    resumeId: string,
    jdId: string,
    weights: { semantic: number; keyword: number; years: number } = {
      semantic: 0.6,
      keyword: 0.95,
      years: 0.05
    }
  ): Promise<MatchResult> {
    // Verify documents exist and are indexed
    const resumeDoc = await db.getDocument(resumeId);
    const jdDoc = await db.getDocument(jdId);
    
    if (!resumeDoc) {
      throw new Error(`Resume document ${resumeId} not found in database`);
    }
    if (!jdDoc) {
      throw new Error(`JD document ${jdId} not found in database`);
    }
    
    if (resumeDoc.status !== 'indexed') {
      throw new Error(`Resume ${resumeId} is not indexed yet. Status: ${resumeDoc.status}`);
    }
    if (jdDoc.status !== 'indexed') {
      throw new Error(`JD ${jdId} is not indexed yet. Status: ${jdDoc.status}`);
    }
    
    if (this.indexMetric === 'cosine') {
      try {
        const detectedMetric = await this.pineconeService.getIndexMetric();
        this.indexMetric = detectedMetric;
      } catch (error) {
        this.indexMetric = 'cosine';
      }
    }

    const jdVector = await this.getJDVector(jdId);
    const jdVector = await this.getJDVector(jdId);

    let allMatches: Array<{ id: string; score: number; metadata: any }> = [];
    
    const largeTopK = 2000;
    const queryFilter1 = {
      doc_id: { $eq: resumeId },
      doc_type: { $eq: 'resume' },
      chunk_index: { $gte: 0 }
    };
    
    try {
      allMatches = await this.pineconeService.query(jdVector, largeTopK, queryFilter1);
    } catch (error) {
      // If filter format 1 fails, try simpler format
      const queryFilter2 = {
        doc_id: resumeId,
        doc_type: 'resume',
        chunk_index: { $gte: 0 }
      };
      try {
        allMatches = await this.pineconeService.query(jdVector, largeTopK, queryFilter2);
      } catch (err) {
      }
    }
    
    if (allMatches.length === 0) {
      try {
        const resumeDoc = await db.getDocument(resumeId);
        const expectedChunkCount = (resumeDoc?.vectorCount || 200) - 1; // Subtract 1 for docvec
        const maxChunksToFetch = Math.min(expectedChunkCount, 200);
        const chunkIds = Array.from({ length: maxChunksToFetch }, (_, i) => `${resumeId}:chunk:${i}`);
        
        const fetchedChunks = await this.pineconeService.fetchByIds(chunkIds);
        
        // Compute similarity scores for fetched chunks
        allMatches = fetchedChunks.map(chunk => {
          const cosineSimilarity = this.computeCosineSimilarity(jdVector, chunk.values);
          const normalizedScore = (cosineSimilarity + 1) / 2;
          return {
            id: chunk.id,
            score: normalizedScore,
            metadata: chunk.metadata
          };
        }).sort((a, b) => b.score - a.score);
      } catch (error) {
        const unfilteredMatches = await this.pineconeService.query(jdVector, 5000);
        allMatches = unfilteredMatches.filter(m => 
          m.metadata?.doc_id === resumeId && 
          m.metadata?.doc_type === 'resume' &&
          (m.metadata?.chunk_index ?? -1) >= 0
        );
      }
    }
    
    const normalizedMatches = allMatches.map(match => ({
      ...match,
      score: this.normalizeScore(match.score, this.indexMetric)
    }));

    const semanticScore = this.computeSemanticScore(normalizedMatches);
    
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

    return {
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
  }

  private async getJDVector(jdId: string): Promise<number[]> {
    const docvecId = `${jdId}:docvec`;
    
    try {
      const fetched = await this.pineconeService.fetchById(docvecId);
      if (fetched && fetched.values && Array.isArray(fetched.values) && fetched.values.length > 0) {
        return fetched.values;
      }
    } catch (error) {
    }

    const jdDoc = await db.getDocument(jdId);
    const jdDoc = await db.getDocument(jdId);
    if (jdDoc && jdDoc.rawText) {
      const vec = await this.embeddingService.embedText(jdDoc.rawText);
      
      try {
        await this.pineconeService.upsertChunks([{
          id: docvecId,
          vector: vec,
          metadata: {
            doc_id: jdId,
            doc_type: 'job_description',
            chunk_index: -1,
            full_text_length: jdDoc.rawText.length
          }
        }]);
      } catch (error) {
      }
      
      return vec;
    }

    throw new Error(`Cannot get JD vector: no docvec and no raw text available for JD: ${jdId}`);
  }

  private computeCosineSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length) {
      throw new Error(`Vector dimension mismatch: ${vec1.length} vs ${vec2.length}`);
    }
    
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }
    
    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    if (denominator === 0) return 0;
    
    return dotProduct / denominator;
  }

  private normalizeScore(rawScore: number, metric: 'cosine' | 'euclidean' | 'dotproduct' = 'cosine'): number {
    if (!Number.isFinite(rawScore)) {
      return 0;
    }

    if (metric === 'cosine') {
      if (rawScore >= 0 && rawScore <= 1) {
        return rawScore;
      }
      return Math.max(0, Math.min(1, (rawScore + 1) / 2));
    } else if (metric === 'dotproduct') {
      return Math.max(0, Math.min(1, rawScore));
    } else {
      return Math.max(0, Math.min(1, 1 - rawScore));
    }
  }

  private computeSemanticScore(matches: Array<{ score: number }>): number {
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
    
    const weights = [0.4, 0.25, 0.15, 0.1, 0.05, 0.025, 0.015, 0.01, 0.005, 0.003];
    let weightedSum = 0;
    let totalWeight = 0;

    for (let i = 0; i < chunksToUse.length; i++) {
      const score = chunksToUse[i];
      const weight = i < weights.length ? weights[i] : Math.exp(-(i - weights.length) * 0.2) * weights[weights.length - 1];
      
      weightedSum += score * weight;
      totalWeight += weight;
    }

    const mean = totalWeight > 0 ? weightedSum / totalWeight : 0;

    const highQualityMatches = chunksToUse.filter(s => s >= 0.7).length;
    const veryHighQualityMatches = chunksToUse.filter(s => s >= 0.85).length;
    const excellentMatches = chunksToUse.filter(s => s >= 0.9).length;
    
    const qualityBoost = Math.min(0.2, 
      highQualityMatches * 0.015 + 
      veryHighQualityMatches * 0.025 + 
      excellentMatches * 0.03
    );
    
    const coverageBoost = Math.min(0.15, Math.min(chunksToUse.length, 15) / 15 * 0.15);

    return Math.max(0, Math.min(1, mean + qualityBoost + coverageBoost));
  }

  private async computeKeywordScore(
    jdId: string,
    resumeId: string,
    topMatches: Array<{ metadata: any }>
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
    const resumeDoc = await db.getDocument(resumeId);
    let resumeText = '';
    
    if (resumeDoc && resumeDoc.rawText) {
      resumeText = resumeDoc.rawText.toLowerCase();
    } else {
      resumeText = topMatches
        .map(m => m.metadata.text_snippet || '')
        .join(' ')
        .toLowerCase();
    }

    const normalizedText = resumeText
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    const matchedSkills: string[] = [];
    const missingSkills: string[] = [];
    const semanticThreshold = 0.65;
    const evidenceThreshold = 0.75;
    const evidenceThresholdWithTextMatch = 0.60;
    const useAgg = 'max';

    for (const skill of allSkills) {
      const isTop = isTopSkill.has(skill);
      
      const semanticMatch = semanticMatches[skill];
      const keywordMatch = this.matchSkill(skill, normalizedText);
      
      const synonyms = skillSynonyms[skill] || [skill];
      
      let semanticMatchWithEvidence = false;
      if (semanticMatch && semanticMatch.evidenceChunks.length > 0) {
        const scores = semanticMatch.evidenceChunks.map(chunk => chunk.score);
        const aggScore = useAgg === 'max' ? Math.max(...scores) : this.meanTopN(scores, 3);
        
        let bestMatchingChunk: { chunk: typeof semanticMatch.evidenceChunks[0]; hasTextMatch: boolean } | null = null;
        let hasTextMatchInAnyChunk = false;
        
        for (const chunk of semanticMatch.evidenceChunks) {
          const hasTextMatch = this.evidenceContainsSkillOrSynonym(chunk.snippet, skill, synonyms);
          
          if (hasTextMatch) {
            hasTextMatchInAnyChunk = true;
            if (!bestMatchingChunk || chunk.score > bestMatchingChunk.chunk.score) {
              bestMatchingChunk = { chunk, hasTextMatch: true };
            }
          }
        }
        
        const effectiveThreshold = hasTextMatchInAnyChunk ? evidenceThresholdWithTextMatch : evidenceThreshold;
        
        const topChunk = semanticMatch.evidenceChunks[0];
        const chunkToCheck = bestMatchingChunk ? bestMatchingChunk.chunk : topChunk;
        const evidenceOk = hasTextMatchInAnyChunk && chunkToCheck.score >= effectiveThreshold;
        
        semanticMatchWithEvidence = aggScore >= semanticThreshold && evidenceOk;
      }

      const isMatched = semanticMatchWithEvidence;
      
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
    
    for (const skill of skills) {
      const contextualizedSkill = `Skill: ${skill} - technical expertise and experience`;
      const embedding = await this.embeddingService.embedText(contextualizedSkill);
      embeddings[skill] = embedding;
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
    const topK = 10; // Query top 10 chunks per skill

    for (const skill of skills) {
      const skillEmbedding = skillEmbeddings[skill];
      if (!skillEmbedding) {
        continue;
      }

      try {
        const queryFilter = {
          doc_id: resumeId,
          doc_type: 'resume',
          chunk_index: { $gte: 0 }
        };

        let matches = await this.pineconeService.query(skillEmbedding, topK, queryFilter);
        
        if (matches.length === 0) {
          const allMatches = await this.pineconeService.query(skillEmbedding, 200);
          matches = allMatches.filter(m => 
            m.metadata?.doc_id === resumeId && 
            m.metadata?.doc_type === 'resume' &&
            (m.metadata?.chunk_index ?? -1) >= 0
          ).slice(0, topK);
        }

        if (matches.length > 0) {
          const normalizedScores = matches.map(m => 
            this.normalizeScore(m.score, this.indexMetric)
          );

          const evidenceChunks = matches.slice(0, 10).map(match => ({
            chunkId: match.id,
            snippet: match.metadata?.text_snippet || '',
            score: this.normalizeScore(match.score, this.indexMetric)
          }));

          results[skill] = {
            similarity: Math.max(...normalizedScores),
            evidenceChunks
          };
        } else {
          results[skill] = {
            similarity: 0,
            evidenceChunks: []
          };
        }
      } catch (error) {
        results[skill] = {
          similarity: 0,
          evidenceChunks: []
        };
      }
    }

    return results;
  }

  private matchSkill(skill: string, text: string): boolean {
    const skillLower = skill.toLowerCase().trim();
    
    if (text.includes(skillLower)) {
      return true;
    }

    const wordBoundaryPattern = new RegExp(`\\b${this.escapeRegex(skillLower)}\\b`, 'i');
    if (wordBoundaryPattern.test(text)) {
      return true;
    }

    const normalizedSkill = this.normalizeSkillName(skillLower);
    const normalizedText = this.normalizeSkillName(text);
    if (normalizedText.includes(normalizedSkill)) {
      return true;
    }

    const variations = this.getSkillVariations(skillLower);
    for (const variation of variations) {
      if (text.includes(variation)) {
        return true;
      }
      const variationPattern = new RegExp(`\\b${this.escapeRegex(variation)}\\b`, 'i');
      if (variationPattern.test(text)) {
        return true;
      }
      const normalizedVariation = this.normalizeSkillName(variation);
      if (normalizedText.includes(normalizedVariation)) {
        return true;
      }
    }

    const words = skillLower.split(/[\s\-_]+/).filter(w => w.length > 2);
    if (words.length > 1) {
      const allWordsPresent = words.every(word => {
        const wordPattern = new RegExp(`\\b${this.escapeRegex(word)}\\b`, 'i');
        return wordPattern.test(text) || normalizedText.includes(this.normalizeSkillName(word));
      });
      if (allWordsPresent) {
        return true;
      }
    }

    if (this.fuzzyMatch(skillLower, text)) {
      return true;
    }

    const acronyms = this.getAcronymExpansions(skillLower);
    for (const expansion of acronyms) {
      if (text.includes(expansion) || normalizedText.includes(this.normalizeSkillName(expansion))) {
        return true;
      }
    }
    const contextPatterns = [
      new RegExp(`(?:experienced|proficient|skilled|expert|knowledge|familiar|worked|used|utilized|implemented|developed|built|created|designed|wrote|programmed|code|coding|programming|development|developing)\\s+(?:with|in|using|on|for|via|through|by)\\s+${this.escapeRegex(skillLower)}`, 'i'),
      new RegExp(`${this.escapeRegex(skillLower)}\\s+(?:experience|expertise|knowledge|proficiency|skills|development|programming|coding)`, 'i'),
    ];
    for (const pattern of contextPatterns) {
      if (pattern.test(text)) {
        return true;
      }
    }

    return false;
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

  private fuzzyMatch(skill: string, text: string): boolean {
    const words = text.toLowerCase().split(/\s+/);
    const skillWords = skill.toLowerCase().split(/\s+/);
    
    if (skillWords.length === 1) {
      const skillWord = skillWords[0];
      for (const word of words) {
        if (word.length < 3) continue;
        
        const distance = this.levenshteinDistance(skillWord, word);
        const maxLength = Math.max(skillWord.length, word.length);
        const similarity = 1 - (distance / maxLength);
        
        if (similarity > 0.85) {
          return true;
        }
      }
    }
    
    const normalizedSkill = this.normalizeSkillName(skill);
    const normalizedText = this.normalizeSkillName(text);
    
    if (normalizedSkill.length > 0 && normalizedText.length > 0) {
      const distance = this.levenshteinDistance(normalizedSkill, normalizedText.substring(0, normalizedSkill.length + 10));
      const maxLength = Math.max(normalizedSkill.length, normalizedSkill.length + 10);
      const similarity = 1 - (distance / maxLength);
      
      if (similarity > 0.85) {
        return true;
      }
    }
    
    return false;
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix: number[][] = [];
    const len1 = str1.length;
    const len2 = str2.length;

    if (len1 === 0) return len2;
    if (len2 === 0) return len1;

    for (let i = 0; i <= len1; i++) {
      matrix[i] = [i];
    }
    for (let j = 0; j <= len2; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= len1; i++) {
      for (let j = 1; j <= len2; j++) {
        if (str1[i - 1] === str2[j - 1]) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j - 1] + 1
          );
        }
      }
    }

    return matrix[len1][len2];
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

  private meanTopN(scores: number[], n: number): number {
    if (scores.length === 0) return 0;
    const topN = scores.slice(0, n);
    return topN.reduce((sum, score) => sum + score, 0) / topN.length;
  }

  private getAcronymExpansions(skill: string): string[] {
    const expansions: string[] = [];
    const skillLower = skill.toLowerCase();
    
    const acronymMap: { [key: string]: string[] } = {
      'ml': ['machine learning', 'machinelearning'],
      'ai': ['artificial intelligence', 'artificialintelligence'],
      'nlp': ['natural language processing', 'naturallanguageprocessing'],
      'cv': ['computer vision', 'computervision'],
      'dl': ['deep learning', 'deeplearning'],
      'ds': ['data science', 'datascience'],
      'bi': ['business intelligence', 'businessintelligence'],
      'etl': ['extract transform load', 'extracttransformload'],
      'api': ['application programming interface', 'applicationprogramminginterface'],
      'rest': ['representational state transfer', 'representationalstatetransfer'],
      'graphql': ['graph query language', 'graphquerylanguage'],
      'sql': ['structured query language', 'structuredquerylanguage'],
      'nosql': ['not only sql', 'notonlysql'],
      'ci/cd': ['continuous integration continuous deployment', 'continuousintegrationcontinuousdeployment'],
      'devops': ['development operations', 'developmentoperations'],
      'aws': ['amazon web services', 'amazonwebservices'],
      'gcp': ['google cloud platform', 'googlecloudplatform'],
      'azure': ['microsoft azure', 'microsoftazure'],
      'ui': ['user interface', 'userinterface'],
      'ux': ['user experience', 'userexperience'],
      'sass': ['software as a service', 'softwareasaservice'],
      'paas': ['platform as a service', 'platformasaservice'],
      'iaas': ['infrastructure as a service', 'infrastructureasaservice'],
    };
    
    if (acronymMap[skillLower]) {
      expansions.push(...acronymMap[skillLower]);
    }
    
    for (const [acronym, exp] of Object.entries(acronymMap)) {
      if (skillLower.includes(acronym)) {
        expansions.push(...exp);
      }
    }
    
    return expansions;
  }

  private getSkillVariations(skill: string): string[] {
    const variations: string[] = [];
    const skillLower = skill.toLowerCase().trim();
    
    const techVariations: { [key: string]: string[] } = {
      // JavaScript ecosystem
      'node.js': ['nodejs', 'node', 'node js', 'nodejs', 'node.js'],
      'react.js': ['reactjs', 'react', 'react js', 'reactjs', 'react.js', 'reactjs'],
      'vue.js': ['vuejs', 'vue', 'vue js', 'vuejs', 'vue.js'],
      'angular.js': ['angularjs', 'angular', 'angular js', 'angularjs', 'angular.js'],
      'next.js': ['nextjs', 'next', 'next js', 'nextjs', 'next.js'],
      'nuxt.js': ['nuxtjs', 'nuxt', 'nuxt js', 'nuxtjs', 'nuxt.js'],
      'express.js': ['express', 'expressjs', 'express js', 'expressjs', 'express.js'],
      'javascript': ['js', 'ecmascript', 'ecma script', 'javascript', 'js'],
      'typescript': ['ts', 'typescript'],
      
      // Databases
      'postgresql': ['postgres', 'postgresql', 'pg', 'postgres db', 'postgresql database'],
      'mongodb': ['mongo', 'mongo db', 'mongodb', 'mongo database'],
      'mysql': ['mysql', 'my sql', 'mysql database'],
      'redis': ['redis', 'redis cache', 'redis database'],
      'elasticsearch': ['elastic search', 'elasticsearch', 'es'],
      'cassandra': ['cassandra', 'apache cassandra'],
      'dynamodb': ['dynamo db', 'dynamodb', 'dynamo database'],
      
      // Programming languages
      'c++': ['cpp', 'c plus plus', 'cplusplus', 'c++'],
      'c#': ['csharp', 'c sharp', 'c#', 'csharp'],
      'python': ['python', 'py', 'python3', 'python 3'],
      'java': ['java', 'java programming', 'java language'],
      'go': ['golang', 'go', 'go language', 'go programming'],
      'rust': ['rust', 'rust language', 'rust programming'],
      'php': ['php', 'php programming'],
      'ruby': ['ruby', 'ruby on rails', 'ruby programming'],
      'swift': ['swift', 'swift programming', 'swift language'],
      'kotlin': ['kotlin', 'kotlin programming'],
      'scala': ['scala', 'scala programming'],
      
      // Frameworks and libraries
      '.net': ['dotnet', 'dot net', 'asp.net', 'asp net', '.net framework', 'dotnet framework'],
      'django': ['django', 'django framework'],
      'flask': ['flask', 'flask framework'],
      'spring': ['spring', 'spring framework', 'spring boot', 'springboot'],
      'laravel': ['laravel', 'laravel framework'],
      'rails': ['ruby on rails', 'rails', 'ror'],
      
      // Cloud and DevOps
      'aws': ['amazon web services', 'amazonwebservices', 'aws cloud'],
      'azure': ['microsoft azure', 'microsoftazure', 'azure cloud'],
      'gcp': ['google cloud platform', 'googlecloudplatform', 'gcp cloud', 'google cloud'],
      'docker': ['docker', 'docker container', 'docker containers'],
      'kubernetes': ['k8s', 'kubernetes', 'k8s orchestration'],
      'terraform': ['terraform', 'terraform infrastructure'],
      'ansible': ['ansible', 'ansible automation'],
      
      // AI/ML
      'machine learning': ['ml', 'machinelearning', 'machine learning', 'ml algorithms'],
      'artificial intelligence': ['ai', 'artificialintelligence', 'artificial intelligence'],
      'deep learning': ['dl', 'deeplearning', 'deep learning', 'neural networks'],
      'natural language processing': ['nlp', 'naturallanguageprocessing', 'natural language processing'],
      'computer vision': ['cv', 'computervision', 'computer vision', 'image processing'],
      'data science': ['ds', 'datascience', 'data science', 'data analytics'],
      
      // Tools and platforms
      'git': ['git', 'git version control', 'git scm'],
      'jenkins': ['jenkins', 'jenkins ci', 'jenkins cicd'],
      'github': ['github', 'github actions', 'github ci'],
      'gitlab': ['gitlab', 'gitlab ci', 'gitlab cicd'],
      'jira': ['jira', 'atlassian jira'],
      'confluence': ['confluence', 'atlassian confluence'],
      
      // Frontend
      'html': ['html', 'html5', 'html 5'],
      'css': ['css', 'css3', 'css 3'],
      'sass': ['sass', 'scss', 'sass css'],
      'less': ['less', 'less css'],
      'webpack': ['webpack', 'webpack bundler'],
      'babel': ['babel', 'babel js', 'babeljs'],
      
      // Testing
      'jest': ['jest', 'jest testing'],
      'mocha': ['mocha', 'mocha testing'],
      'cypress': ['cypress', 'cypress testing'],
      'selenium': ['selenium', 'selenium webdriver'],
      
      // Mobile
      'react native': ['reactnative', 'react native', 'reactnative'],
      'flutter': ['flutter', 'flutter development'],
      'ios': ['ios', 'ios development', 'apple ios'],
      'android': ['android', 'android development', 'google android'],
    };

    for (const [key, vars] of Object.entries(techVariations)) {
      const normalizedKey = this.normalizeSkillName(key);
      const normalizedSkill = this.normalizeSkillName(skillLower);
      
      if (skillLower.includes(key) || key.includes(skillLower) || 
          normalizedSkill.includes(normalizedKey) || normalizedKey.includes(normalizedSkill)) {
        variations.push(...vars);
      }
    }

    const versionMatch = skillLower.match(/^(.+?)\s*(?:v|version)?\s*(\d+(?:\.\d+)?)$/);
    if (versionMatch) {
      const baseSkill = versionMatch[1].trim();
      const version = versionMatch[2];
      variations.push(baseSkill);
      variations.push(baseSkill + version);
      variations.push(baseSkill + ' ' + version);
      variations.push(baseSkill + 'v' + version);
    }

    const commonSuffixes = ['framework', 'library', 'tool', 'technology', 'platform', 'service'];
    for (const suffix of commonSuffixes) {
      if (skillLower.endsWith(' ' + suffix)) {
        variations.push(skillLower.replace(' ' + suffix, ''));
      }
      if (!skillLower.includes(suffix)) {
        variations.push(skillLower + ' ' + suffix);
      }
    }

    return [...new Set(variations)];
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

      for (let i = 0; i < rangeMatches.length; i++) {
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
      for (let i = 0; i < monthNames.length; i++) {
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
    for (let i = 0; i < monthNames.length; i++) {
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

