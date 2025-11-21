import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';

export interface ExtractionResult {
  text: string;
  pages: number;
}

export class TextExtractionService {
  async extractText(buffer: Buffer, mimetype: string, filename: string): Promise<ExtractionResult> {
    if (mimetype === 'application/pdf' || filename.endsWith('.pdf')) {
      return this.extractFromPDF(buffer);
    } else if (
      mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
      mimetype === 'application/msword' ||
      filename.endsWith('.docx') ||
      filename.endsWith('.doc')
    ) {
      return this.extractFromDOCX(buffer);
    } else if (mimetype === 'text/plain' || filename.endsWith('.txt')) {
      return this.extractFromText(buffer);
    } else {
      throw new Error(`Unsupported file type: ${mimetype}`);
    }
  }

  private async extractFromPDF(buffer: Buffer): Promise<ExtractionResult> {
    try {
      const data = await pdfParse(buffer);
      return {
        text: data.text,
        pages: data.numpages
      };
    } catch (error) {
      throw new Error(`Failed to extract text from PDF: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async extractFromDOCX(buffer: Buffer): Promise<ExtractionResult> {
    try {
      const result = await mammoth.extractRawText({ buffer });
      const wordCount = result.value.split(/\s+/).length;
      const estimatedPages = Math.max(1, Math.ceil(wordCount / 500));
      
      return {
        text: result.value,
        pages: estimatedPages
      };
    } catch (error) {
      throw new Error(`Failed to extract text from DOCX: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async extractFromText(buffer: Buffer): Promise<ExtractionResult> {
    const text = buffer.toString('utf-8');
    const wordCount = text.split(/\s+/).length;
    const estimatedPages = Math.max(1, Math.ceil(wordCount / 500));
    
    return {
      text,
      pages: estimatedPages
    };
  }
}

