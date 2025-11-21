import { GoogleGenerativeAI } from '@google/generative-ai';

export interface ExtractedSkills {
  topSkills: string[]; // 4-5 most critical skills based on market standards
  generalSkills: string[]; // Remaining skills
  requiredYears?: number;
  skillSynonyms?: { [skill: string]: string[] }; // LLM-generated synonyms for each skill
}

export class JDSkillExtractionService {
  private genAI: GoogleGenerativeAI;

  constructor(apiKey?: string) {
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY is required for skill extraction');
    }
    this.genAI = new GoogleGenerativeAI(apiKey);
  }

  async extractSkills(jdText: string): Promise<ExtractedSkills> {
    const prompt = `Analyze the following Job Description and extract technical skills. Based on current market standards and industry best practices, identify the 4-5 most critical and essential skills that are absolutely required for this role. These top skills should be the core competencies that define the position. All other relevant skills should be categorized as general skills.

For each skill (both top and general), provide 3-5 realistic synonyms/variations that are commonly seen in resumes. The response should be a JSON object where keys are skill names and values are arrays containing the skill itself plus its synonyms.

Return strict JSON: { 
  "topSkills": ["skill1","skill2","skill3","skill4","skill5"], 
  "generalSkills": ["skill6","skill7","skill8"], 
  "requiredYears": 3,
  "skillSynonyms": {
    "skill1": ["skill1", "synonym1", "synonym2", "synonym3"],
    "skill2": ["skill2", "synonym1", "synonym2"],
    "skill3": ["skill3", "synonym1", "synonym2", "synonym3"],
    "skill6": ["skill6", "synonym1", "synonym2"],
    "skill7": ["skill7", "synonym1", "synonym2", "synonym3"]
  }
}

Guidelines:
- topSkills: Select 4-5 skills that are the most critical based on market standards. These should be the core technical competencies.
- generalSkills: Include all other relevant technical skills mentioned in the JD.
- requiredYears: Extract the number of years of experience required (if mentioned).
- skillSynonyms: For EACH skill (both top and general), provide an array where the first element is the skill name itself, followed by 3-5 realistic synonyms/variations commonly seen in resumes. Include variations like different spellings, abbreviations, and alternative names.

Example skillSynonyms format:
{
  "node.js": ["node.js", "nodejs", "node", "server-side javascript"],
  "react": ["react", "react.js", "reactjs", "next.js"],
  "aws": ["aws", "amazon web services", "ec2", "s3"],
  "microservices": ["microservices", "micro-service", "micro service", "distributed services"]
}

Job Description:

---

${jdText}`;

    try {
      const model = this.genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
      const result = await model.generateContent(prompt);
      const response = result.response;
      const text = response.text();

      // Extract JSON from response (handle markdown code blocks if present)
      let jsonText = text.trim();
      if (jsonText.startsWith('```json')) {
        jsonText = jsonText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      } else if (jsonText.startsWith('```')) {
        jsonText = jsonText.replace(/```\n?/g, '').trim();
      }

      const parsed = JSON.parse(jsonText) as ExtractedSkills;

      const topSkills = Array.isArray(parsed.topSkills) ? parsed.topSkills : [];
      const generalSkills = Array.isArray(parsed.generalSkills) ? parsed.generalSkills : [];
      
      const normalizedTopSkills = topSkills.slice(0, 5);
      
      const allSkills = [...normalizedTopSkills, ...generalSkills];
      const skillSynonyms: { [skill: string]: string[] } = {};
      
      if (parsed.skillSynonyms && typeof parsed.skillSynonyms === 'object') {
        for (const skill of allSkills) {
          if (parsed.skillSynonyms[skill] && Array.isArray(parsed.skillSynonyms[skill])) {
            const synonymsArray = parsed.skillSynonyms[skill];
            const uniqueSynonyms = [...new Set(synonymsArray)];
            if (!uniqueSynonyms.includes(skill)) {
              skillSynonyms[skill] = [skill, ...uniqueSynonyms].slice(0, 6);
            } else {
            const skillIndex = uniqueSynonyms.indexOf(skill);
            const reordered = [skill, ...uniqueSynonyms.filter((_, i) => i !== skillIndex)];
            skillSynonyms[skill] = reordered.slice(0, 6);
            }
          } else {
            skillSynonyms[skill] = [skill];
          }
        }
      } else {
        for (const skill of allSkills) {
          skillSynonyms[skill] = [skill];
        }
      }
      
      return {
        topSkills: normalizedTopSkills,
        generalSkills: generalSkills,
        requiredYears: typeof parsed.requiredYears === 'number' ? parsed.requiredYears : undefined,
        skillSynonyms: skillSynonyms
      };
    } catch (error) {
      return {
        topSkills: [],
        generalSkills: [],
        requiredYears: undefined,
        skillSynonyms: {}
      };
    }
  }

}

