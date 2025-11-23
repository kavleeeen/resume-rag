import { Storage } from '@google-cloud/storage';
import dotenv from 'dotenv';

dotenv.config();

export interface UploadResult {
  filename: string;
  originalName: string;
  url: string;
  longTermUrl: string;
  size: number;
  mimetype: string;
  expiresAt: string;
  longTermExpiresAt: string;
}

export class GCStorageService {
  private storage: Storage;
  private bucket: any;

  constructor() {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT_ID || 'acquired-voice-474914-j2';
    const bucketName = process.env.GCS_BUCKET;

    if (!bucketName) {
      throw new Error('GCS_BUCKET environment variable is not set');
    }

    const credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
    if (credentialsPath) {
      this.storage = new Storage({
        projectId,
        keyFilename: credentialsPath
      });
    } else {
      this.storage = new Storage({
        projectId
      });
    }

    this.bucket = this.storage.bucket(bucketName);
  }

  async uploadFile(
    buffer: Buffer,
    originalName: string,
    mimetype: string
  ): Promise<UploadResult> {
    const timestamp = Date.now();
    const safeName = `${timestamp}_${originalName.replace(/\s+/g, '_')}`;

    const file = this.bucket.file(safeName);

    await new Promise<void>((resolve, reject) => {
      const stream = file.createWriteStream({
        metadata: {
          contentType: mimetype
        },
        resumable: false
      });

      stream.on('error', (err: Error) => {
        reject(new Error(`GCS upload failed: ${err.message}`));
      });

      stream.on('finish', () => {
        resolve();
      });

      stream.end(buffer);
    });

    const [signedUrl] = await file.getSignedUrl({
      action: 'read',
      expires: Date.now() + 60 * 60 * 1000,
      version: 'v4'
    });

    const [longTermSignedUrl] = await file.getSignedUrl({
      action: 'read',
      expires: Date.now() + 7 * 24 * 60 * 60 * 1000,
      version: 'v4'
    });

    return {
      filename: safeName,
      originalName,
      url: signedUrl,
      longTermUrl: longTermSignedUrl,
      size: buffer.length,
      mimetype,
      expiresAt: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
      longTermExpiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString()
    };
  }
}

