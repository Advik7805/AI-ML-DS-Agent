/**
 * Turn a provider failure into one readable sentence — and keep a stack trace out of the UI.
 *
 * The failure this exists to remove: Groq retired the model the hub was pinned to, and the
 * upload studio rendered
 *
 *     RuntimeError: Groq 404: {"error":{"message":"The model llama-3.3-70b-versatile does not
 *     exist or you do not have access to it.","type":"invalid_request_error","code":"model_not_found"}}
 *
 * *in the box where the answer belongs*, on a page a first-time visitor was reading. That string
 * is four layers of noise — an exception class, an HTTP token, a JSON envelope, and one sentence
 * that actually means something — and the answer rendered none of the meaning and all of the
 * noise.
 *
 * So: dig out the innermost sentence, decide which of the handful of real failures it is, and
 * lead with what the reader can do about it. The raw text survives under "Underneath:", because
 * hiding an error entirely is its own kind of dishonesty — this is for a person who wants to
 * know what happened, not for one who should be shielded from it.
 *
 * Deliberately total: every input, including nonsense, returns a string. A blank where an
 * explanation should be is the failure mode this whole project keeps trying to avoid.
 */

/** The model itself is gone or unreachable. Matches Groq's `model_not_found` and Gemini's 404. */
const MODEL_RETIRED = /model_not_found|does not exist|decommission|deprecated|unknown model|invalid model/i;
/** Quota or rate limit — the request was fine, the account's allowance was not. */
const RATE_LIMIT = /rate[\s_-]?limit|too many requests|429|quota|resource[\s_-]?exhausted/i;
/** The key was rejected. Retrying the same key will not fix it. */
const BAD_KEY =
  /invalid api key|api[\s_-]?key[\s_-]?invalid|api key not valid|unauthorized|401|403|permission denied/i;
/** Nothing came back in time — a sleeping free-tier instance is the usual cause. */
const TIMEOUT = /timeout|timed out|deadline exceeded|etimedout|readtimeout/i;
/** The provider answered, but with "not now". */
const UNAVAILABLE = /503|502|504|unavailable|overloaded|bad gateway|service is currently/i;

/** `RuntimeError:` / `ValueError:` style prefixes, which add nothing a reader needs. */
const PREFIX = /^[A-Za-z_][\w.]*(?:Error|Exception)\s*:\s*/;

/** Whichever key a vendor used for "here is what actually went wrong". */
function messageOf(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  const error = record.error;
  if (error && typeof error === "object") {
    const inner = (error as Record<string, unknown>).message;
    if (typeof inner === "string" && inner.trim()) return inner.trim();
  }
  if (typeof error === "string" && error.trim()) return error.trim();
  for (const key of ["message", "detail"]) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

/** The readable core of a raw error: prefix stripped, JSON envelope unwrapped, whitespace flat. */
function readable(raw: string): string {
  let text = raw.replace(/\s+/g, " ").trim().replace(PREFIX, "");
  const start = text.indexOf("{");
  if (start >= 0) {
    try {
      const message = messageOf(JSON.parse(text.slice(start)));
      // Keep whatever preceded the JSON ("Groq 404:") — it is the only part that says *which*
      // vendor failed, which matters as soon as a deployment has more than one key configured.
      if (message) text = `${text.slice(0, start)}${message}`;
    } catch {
      /* not JSON after all — the flat text is the best available reading */
    }
  }
  return text.replace(PREFIX, "").trim().replace(/[.\s]+$/, "");
}

/**
 * One sentence of guidance plus the vendor's own words. Ends with "Underneath: …".
 *
 * The tests that matter here are in `tests/test_frontend_contract.py`, which asserts against
 * this source: the UI must route `payload.error` through this function rather than rendering it,
 * because the bug being fixed was exactly a raw error reaching a page.
 */
export function readAgentError(raw: string | null | undefined): string {
  if (!raw) return "";
  const text = readable(String(raw));
  const underneath = `Underneath: ${text || "no detail was returned"}.`;

  if (MODEL_RETIRED.test(text)) {
    // Name the variable that actually fixes it — the vendor is in the text we just unwrapped,
    // and "set GROQ_MODEL" is actionable where "set *_MODEL" is a puzzle.
    const envVar = /gemini/i.test(text) ? "GEMINI_MODEL" : /groq/i.test(text) ? "GROQ_MODEL" : "*_MODEL";
    return (
      "The language model this deployment is pointed at has been retired by the provider, so " +
      `the agent could not answer. Setting ${envVar} to a current model fixes it. ${underneath}`
    );
  }
  if (RATE_LIMIT.test(text)) {
    return (
      "The model provider is rate-limiting this deployment, so the agent could not answer. " +
      `Give it a minute and ask again. ${underneath}`
    );
  }
  if (BAD_KEY.test(text)) {
    return (
      "The model provider rejected this deployment's API key, so the agent could not answer. " +
      `The key on the API service needs attention; asking again will not help. ${underneath}`
    );
  }
  if (TIMEOUT.test(text)) {
    return (
      "The model provider did not answer in time, so the agent could not answer. A free-tier " +
      `API waking from sleep is the usual cause — try again in a few seconds. ${underneath}`
    );
  }
  if (UNAVAILABLE.test(text)) {
    return (
      "The model provider is temporarily unavailable, so the agent could not answer. The API " +
      `itself is up; try again shortly. ${underneath}`
    );
  }
  return `The agent could not answer that question. ${underneath}`;
}
