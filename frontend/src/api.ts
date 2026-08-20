/**
 * Typed client for the CineMatch API.
 *
 * The backend returns structured detail on 4xx (`{"detail": "..."}`), so we
 * surface that text rather than a bare status code — a 409 from a deployment
 * without interaction data is a message the user needs to read, not an error
 * to swallow.
 */

import type {
  Evaluation,
  Health,
  Movie,
  RecommendResponse,
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
    // fetch only rejects on network failure, never on a 4xx/5xx.
    throw new ApiError(0, `Cannot reach the API at ${BASE}. Is uvicorn running?`);
  }

  if (!response.ok) {
    let detail = `Request failed with ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) {
        // Pydantic validation errors arrive as a list of objects.
        detail = body.detail.map((e: { msg?: string }) => e.msg).join("; ");
      }
    } catch {
      /* keep the status-code fallback */
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),

  search: (q: string, limit = 20) =>
    request<Movie[]>(
      `/movies?q=${encodeURIComponent(q)}&limit=${limit}`,
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
