import type {
  Evaluation,
  Health,
  Movie,
  RecommendResponse,
  SearchResponse,
  SimilarResponse,
} from "./types";

const BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${BASE}. Is uvicorn running?`);
  }

  if (!response.ok) {
    let detail = `Request failed with ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) {
        detail = body.detail.map((e: { msg?: string }) => e.msg).join("; ");
      }
    } catch {}
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),

  // Title substring match, backing the seed picker: there, matching the
  // literal characters the user is typing is the correct behaviour.
  titleSearch: (q: string, limit = 20) =>
    request<Movie[]>(
      `/movies?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),

  // Meaning-based match over the content embeddings. Returns 503 when the
  // deployment omits artifacts/encoder/, which `Health.has_semantic_search`
  // reports up front so the UI can hide the box instead of failing on submit.
  semanticSearch: (q: string, k = 12) =>
    request<SearchResponse>(
      `/search?q=${encodeURIComponent(q)}&k=${k}`,
    ),

  movie: (movieId: number) => request<Movie>(`/movies/${movieId}`),

  similar: (movieId: number, k = 10) =>
    request<SimilarResponse>(`/movies/${movieId}/similar?k=${k}`),

  recommend: (body: {
    seed_movie_ids?: number[];
    user_id?: number;
    k?: number;
    mmr_lambda?: number;
    exclude_seen?: boolean;
  }) =>
    request<RecommendResponse>("/recommend", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  evaluation: () => request<Evaluation>("/evaluation"),
};

export const API_BASE = BASE;
