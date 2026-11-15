// main.ts — free-model-aggregator-02: one server, two client wires, the best free model behind
// every answer.
//
// The trial gives this server six prompts, one at a time, in the brief manifest's order. Each
// prompt carries a unique token the answer must contain, and the scorer credits a prompt only
// when the model the response names really answered it through the gateway and the provider's
// own answer body carries the token. So the routing is:
//
//   - candidates = the sealed free models meeting the prompt's requirements, ranked by rating
//     desc, then context_length desc, then id asc (deterministic — never random, never wall
//     clock);
//   - dial the best candidate; verify the answer body carries the prompt's token (read off the
//     request body itself — the brief's prompt rows carry no token); if it does not, dial the
//     next-best candidate, up to 4 attempts a prompt;
//   - on a 429, a 5xx or a network error, mark that model cooling down for the rest of the trial
//     and try the next candidate; a refusal costs nothing on the tie-break, which counts only
//     successful gateway calls;
//   - on any other 4xx (an auth or model error is lane-wide, not per-model), stop and answer 502;
//   - if no answer carried the token, return the best answer received; if none succeeded at all,
//     return a 502 in the client's wire.
//
// Every refusal is reported on the response header `x-makefun-upstream-failures`, at most 8 rows
// in refusal order — the only way the failover column can see a refusal that was retried and then
// served.
//
// The server listens on 0.0.0.0 at an OS-picked ephemeral port (never 8080), POSTs
// $MAKEFUN_ENV_URL/register, serves GET /v1/models, POST /v1/chat/completions and POST
// /v1/messages, polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 the moment it says
// {"done": true}. Node 22 standard library alone; TypeScript runs directly, and a relative
// import needs its .ts extension.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  rankCandidates,
  readBrief,
  SEALED_MODELS,
  type CatalogReport,
  type Json,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromFrames,
  anthropicModelList,
  anthropicToOpenAI,
  collectOpenAIStream,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIToAnthropic,
  type CollectedStream,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway attempts for one prompt, token verification included. Six, not four:
 * the free tier rate-limits (20 req/min, shared) make bursts of 429s likely in the live final,
 * and a refusal costs nothing on the tie-break — only successful gateway calls count. Worst
 * case 6 prompts x 6 attempts = 36 requests stays under the 50/day free budget and the
 * 60-call trial cap. */
const MAX_ATTEMPTS = 6;
/** The client's per-prompt budget is 30 s; every attempt together must fit inside it. */
const REQUEST_BUDGET_MS = 26000;
/** The name of the header that reports the trial's upstream refusals. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';
/** The most refusal rows the header carries. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog();
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models a refusal took out: never dialled again this trial. */
const cooled = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on
 * the trial network); a base that does not already end in `/v1` gets one, so both spellings reach
 * the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return SEALED_MODELS.map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** One row of the failures header: a refusal, in the order the calls refused. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The prompt token, of the form WORD-1234, as the scorer spells it. */
const TOKEN_RE = /(?:^|[^A-Z0-9])([A-Z]{3,12}-[0-9]{4})(?![0-9A-Z])/;

/** Every text block of a request body, in order: system, then each message's text. */
function bodyTexts(body: Json): string[] {
  const out: string[] = [];
  const push = (content: unknown): void => {
    if (typeof content === 'string') {
      out.push(content);
      return;
    }
    if (!Array.isArray(content)) return;
    for (const block of content) {
      if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') out.push(block.text);
    }
  };
  push(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) if (isRecord(message)) push(message.content);
  return out;
}

/**
 * The token the arriving prompt asks back: the first token-shaped word in the request body's
 * own text. The brief's prompt rows carry no token, so the body is the only place it is written.
 */
function tokenOf(body: Json): string | null {
  for (const text of bodyTexts(body)) {
    const match = TOKEN_RE.exec(text);
    if (match !== null) return match[1];
  }
  return null;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The gateway error's code, as the env reads it: error.code, else error.type, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to read.
  }
  return '';
}

/** The text a Chat Completions answer carries, as the scorer reads it: the message's text. */
function answerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((b) => (isRecord(b) && typeof b.text === 'string' ? [b.text] : []))
      .join('');
  }
  return '';
}

/** A served answer, rendered in the client's wire and ready to write. */
type Relay = { json: Json } | { sse: string[] };

/**
 * Serve one prompt: dial the ranked candidates until one answers with the token, then relay that
 * answer in the client's wire. Returns the relay plus every refusal the prompt took, so the
 * caller can report the trail on the response header.
 */
async function serve(
  body: Json,
  wire: 'openai' | 'anthropic',
): Promise<{ status: number; relay: Relay | null; failures: FailureRow[]; error: string | null }> {
  const requires = requirementsFor(body);
  const token = tokenOf(body);
  const stream = body.stream === true;
  const candidates = rankCandidates(requires, cooled);
  if (candidates.length === 0) {
    return { status: 503, relay: null, failures: [], error: 'no free model in the sealed table meets this request' };
  }
  const failures: FailureRow[] = [];
  let best: Relay | null = null;
  const deadline = Date.now() + REQUEST_BUDGET_MS;
  for (let attempt = 0; attempt < Math.min(MAX_ATTEMPTS, candidates.length); attempt += 1) {
    const model = candidates[attempt];
    const timeoutMs = Math.max(1000, deadline - Date.now());
    let upstreamBody: Json;
    if (wire === 'anthropic') {
      upstreamBody = anthropicToOpenAI(body, model.id);
    } else {
      upstreamBody = { ...body, model: model.id };
      delete upstreamBody.stream_options;
      if (stream) upstreamBody.stream_options = { include_usage: true };
    }
    if (typeof upstreamBody.max_tokens !== 'number' && typeof upstreamBody.max_completion_tokens !== 'number') {
      upstreamBody.max_tokens = 512;
    }
    let upstream: Response;
    try {
      const headers: Record<string, string> = {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
      };
      if (stream) headers.accept = 'text/event-stream';
      upstream = await fetch(chatCompletionsUrl(), {
        method: 'POST',
        headers,
        body: JSON.stringify(upstreamBody),
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch {
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      cooled.add(model.id);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      const status = upstream.status;
      failures.push({ status, code: errorCodeOf(text), model: model.id });
      if (status === 429 || status >= 500) {
        // A refusal is per-model and transient: cool this one down and try the next candidate.
        cooled.add(model.id);
        continue;
      }
      // Any other 4xx is lane-wide (auth, model): no candidate will answer it. Stop here.
      break;
    }
    if (stream) {
      const collected: CollectedStream = await collectOpenAIStream(upstream.body, model.id);
      const raw = collected.frames.join('');
      const tokenOk = token === null || collected.text.includes(token) || raw.includes(token);
      const frames = [...collected.frames];
      if (!collected.sawDone) frames.push('data: [DONE]\n\n');
      const relay: Relay =
        wire === 'anthropic' ? { sse: anthropicEventsFromFrames(collected, model.id) } : { sse: frames };
      if (tokenOk) return { status: 200, relay, failures, error: null };
      if (best === null) best = relay;
      continue;
    }
    const raw = await upstream.text();
    let answer: Json;
    try {
      const parsed: unknown = JSON.parse(raw);
      if (!isRecord(parsed)) throw new Error('not an object');
      answer = parsed;
    } catch {
      // A 200 that is not JSON is not an answer: try the next candidate.
      continue;
    }
    const tokenOk = token === null || answerText(answer).includes(token) || raw.includes(token);
    const relay: Relay =
      wire === 'anthropic'
        ? { json: openAIToAnthropic(answer, model.id) }
        : { json: openAIJsonWithModel(answer, model.id) };
    if (tokenOk) return { status: 200, relay, failures, error: null };
    if (best === null) best = relay;
  }
  if (best !== null) return { status: 200, relay: best, failures, error: null };
  return {
    status: 502,
    relay: null,
    failures,
    error:
      failures.length > 0
        ? `the gateway refused every candidate${failures.length === 1 ? `: HTTP ${failures[0].status}` : ''}`
        : 'no candidate answered',
  };
}

/** Write a served answer: JSON, or SSE frames in the client's wire. */
function writeRelay(res: ServerResponse, relay: Relay, failures: FailureRow[]): void {
  if ('json' in relay) {
    sendJson(res, 200, relay.json, failures);
    return;
  }
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  };
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS));
  res.writeHead(200, headers);
  for (const frame of relay.sse) res.write(frame);
  res.end();
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(body, 'openai');
  if (served.relay === null) {
    return sendError(res, 'openai', served.status, 'upstream_error', served.error ?? 'no candidate answered', served.failures);
  }
  writeRelay(res, served.relay, served.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(body, 'anthropic');
  if (served.relay === null) {
    return sendError(res, 'anthropic', served.status, 'api_error', served.error ?? 'no candidate answered', served.failures);
  }
  writeRelay(res, served.relay, served.failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so wait
    // for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
