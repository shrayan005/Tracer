// One fetch wrapper for every API call: JSON in, JSON out, and errors as ApiError with the HTTP status,
// the backend's message ("detail") and Retry-After when the backend sends it (429).

import { readSse } from "../lib/sse";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public retryAfter?: number,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin", // the login cookie (httpOnly: JavaScript never sees it)
    ...init,
    headers: { "content-type": "application/json", ...init.headers },
  });
  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response),
                       Number(response.headers.get("retry-after")) || undefined);
  }
  if (response.status === 204) return undefined as T;
  const type = response.headers.get("content-type") ?? "";
  return (type.includes("json") ? response.json() : response.text()) as Promise<T>;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg; // FastAPI 422
  } catch {
    /* not JSON */
  }
  return response.statusText || `HTTP ${response.status}`;
}

export const post = <T>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

/** POST, then read the reply as Server-Sent Events: `onEvent` gets each event (data parsed from JSON) as it
 *  arrives, and the promise resolves with the `done` event's data. Errors before the stream (404, 409, 429)
 *  come as a normal JSON reply; errors during it as an `error` event: both become ApiError. */
export async function postStream<T>(path: string, body: unknown,
                                    onEvent: (event: string, data: any) => void = () => {}): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response),
                       Number(response.headers.get("retry-after")) || undefined);
  }
  let result: { value: T } | null = null;
  let failure: ApiError | null = null;
  await readSse(response, ({ event, data }) => {
    const parsed = JSON.parse(data);
    if (event === "done") result = { value: parsed };
    else if (event === "error") failure = new ApiError(500, parsed.detail ?? "the answer failed");
    else onEvent(event, parsed);
  });
  if (failure) throw failure;
  if (!result) throw new ApiError(502, "the connection closed before the answer was finished");
  return (result as { value: T }).value;
}
