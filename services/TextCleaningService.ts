export class TextCleaningService {
  cleanText(text: string): string {
    let cleaned = text;

    cleaned = cleaned.replace(/\r\n/g, '\n');
    cleaned = cleaned.replace(/\r/g, '\n');
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    cleaned = cleaned.replace(/[ \t]+/g, ' ');
    cleaned = cleaned.replace(/[ \t]+$/gm, '');

    cleaned = cleaned.replace(/^Page\s+\d+\s+of\s+\d+$/gim, '');
    cleaned = cleaned.replace(/^\d+\s*\/\s*\d+$/gm, '');
    cleaned = cleaned.replace(/^-\s*\d+\s*-$/gm, '');
    cleaned = cleaned.replace(/^\d+$/gm, '');

    cleaned = cleaned.replace(/^.*?Resume.*?$/gim, '');
    cleaned = cleaned.replace(/^.*?Confidential.*?$/gim, '');
    cleaned = cleaned.replace(/^.*?Page.*?$/gim, '');

    cleaned = cleaned.replace(/^[^\s@]+@[^\s@]+\.[^\s@]+$/gm, '');

    cleaned = cleaned.replace(/^https?:\/\/[^\s]+$/gm, '');

    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    cleaned = cleaned.trim();

    return cleaned;
  }
}

