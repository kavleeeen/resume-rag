export interface Section {
  name: string;
  startIndex: number;
  endIndex: number;
  text: string;
}

export class SectionDetectionService {
  private readonly sectionPatterns = [
    /^(Experience|Work History|Work Experience|Professional Experience|Employment History)/i,
    /^(Education|Academic Background|Educational Background)/i,
    /^(Skills|Technical Skills|Core Competencies|Competencies)/i,
    /^(Responsibilities|Key Responsibilities|Job Responsibilities)/i,
    /^(Requirements|Required Skills|Qualifications|Required Qualifications)/i,
    /^(Summary|Professional Summary|Profile|Objective)/i,
    /^(Projects|Project Experience)/i,
    /^(Certifications|Certificates)/i,
    /^(Languages|Language Skills)/i,
    /^(Awards|Honors|Achievements)/i
  ];

  detectSections(text: string): Section[] {
    const sections: Section[] = [];
    const lines = text.split('\n');
    let currentSection: Section | null = null;
    let currentStartIndex = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      const matchedPattern = this.sectionPatterns.find(pattern => pattern.test(line));
      
      if (matchedPattern) {
        if (currentSection) {
          currentSection.endIndex = currentStartIndex;
          currentSection.text = text.substring(currentSection.startIndex, currentSection.endIndex);
          sections.push(currentSection);
        }

        const sectionName = this.extractSectionName(line);
        currentSection = {
          name: sectionName,
          startIndex: currentStartIndex,
          endIndex: text.length,
          text: ''
        };
      }

      currentStartIndex += line.length + 1;
    }

    if (currentSection) {
      currentSection.endIndex = text.length;
      currentSection.text = text.substring(currentSection.startIndex, currentSection.endIndex);
      sections.push(currentSection);
    }

    return sections;
  }

  private extractSectionName(line: string): string {
    const cleaned = line.replace(/[^\w\s]/g, '').trim().toLowerCase();
    
    if (cleaned.includes('experience') || cleaned.includes('work')) return 'experience';
    if (cleaned.includes('education')) return 'education';
    if (cleaned.includes('skill')) return 'skills';
    if (cleaned.includes('responsibilit')) return 'responsibilities';
    if (cleaned.includes('requirement') || cleaned.includes('qualification')) return 'requirements';
    if (cleaned.includes('summary') || cleaned.includes('profile') || cleaned.includes('objective')) return 'summary';
    if (cleaned.includes('project')) return 'projects';
    if (cleaned.includes('certif')) return 'certifications';
    if (cleaned.includes('language')) return 'languages';
    if (cleaned.includes('award') || cleaned.includes('honor') || cleaned.includes('achievement')) return 'awards';
    
    return cleaned;
  }

  getSectionForText(text: string, position: number): string | undefined {
    const sections = this.detectSections(text);
    for (const section of sections) {
      if (position >= section.startIndex && position < section.endIndex) {
        return section.name;
      }
    }
    return undefined;
  }
}

