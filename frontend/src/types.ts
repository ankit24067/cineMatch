/**
 * Mirrors `api/schemas.py`. Kept hand-written rather than generated so the
 * drift is visible in review: if a field here has no counterpart there, one
 * of the two is wrong.
 */

export interface Movie {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  poster: string | null;
  overview: string | null;
}

/** Component contributions are fractions of the final score and sum to 1.0. */
export interface Explanation {
  contributions: Record<string, number>;
  because_of: Movie | null;
  seed_similarity: number;
}

export interface Recommendation {
  movie: Movie;
  score: number;
  rank: number;
  explanation: Explanation;
}

export interface RecommendResponse {
  recommendations: Recommendation[];
  seeds: Movie[];
  personalised: boolean;
  cold_start: boolean;
  took_ms: number;
}

export interface SimilarResponse {
  movie: Movie;
  similar: Movie[];
  similarities: number[];
}

export interface Health {
  status: string;
  n_items: number;
  n_users: number;
  has_interactions: boolean;
  artifacts_built_at: string | null;
  weights: Record<string, number>;
  mmr_lambda: number;
}

export interface EvaluationRow {
  name: string;
  precision: number;
  recall: number;
  ndcg: number;
  map_: number;
  coverage: number;
  novelty: number;
  diversity: number;
  p50_ms: number;
  p95_ms: number;
  n_users: number;
}

export interface Evaluation {
  config: {
    weights: Record<string, number>;
    lambda: number;
    factors: number;
    regularisation: number;
    iterations: number;
    alpha: number;
  };
  results: EvaluationRow[];
  cohorts: Record<string, { n: number; ndcg: number }>;
}
