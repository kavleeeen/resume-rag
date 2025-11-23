/**
 * Utility for normalizing similarity scores across different metrics
 * Follows DRY principle - shared by ChatService and MatchingService
 */
export function normalizeScore(
  rawScore: number,
  metric: 'cosine' | 'euclidean' | 'dotproduct' = 'cosine'
): number {
  if (!Number.isFinite(rawScore)) return 0;

  if (metric === 'cosine') {
    // Tuned for real-world resume → JD similarity
    // Raw cosine mapping: 0.9→0.875, 0.6→0.5, 0.4→0.25, ≤0.2→0.0
    return Math.max(0, Math.min(1, (rawScore - 0.2) / 0.8));
  } else if (metric === 'dotproduct') {
    const DOT_MAX = 40;
    return Math.max(0, Math.min(1, rawScore / DOT_MAX));
  } else {
    // Euclidean distance: convert to similarity
    const MAX_DIST = 10;
    return Math.max(0, Math.min(1, 1 - (rawScore / MAX_DIST)));
  }
}

