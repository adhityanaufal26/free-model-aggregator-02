// lib/providers.ts — the sealed free-model table, the brief reader, the catalog, the ranking.
//
// The model table below is the house's sealed copy of the live OpenRouter free catalog, as the
// kit publishes it: rating in milli-points, what each model can do, its context window. It is
// hardcoded on purpose — the scorer holds the same table sealed, so a brief that disagreed with
// it would be rejected by the scorer before any prompt is served. The brief file is read only
// for the prompt rows (id, api, stream, weight, requirements), in arrival order.
//
// Ranking is deterministic: rating desc, then context_length desc, then id asc. No clock, no
// randomness anywhere in this file.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model. `id` is the id the gateway routes; `bare` is the same without the prefix. */
export interface ModelRow {
  id: string;
  bare: string;
  provider: string;
  display_name: string;
  /** The house's quality rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this entry reads it: prompts only. `error` is set when it could not be read. */
export interface Brief {
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

function row(
  bare: string,
  display_name: string,
  rating_milli: number,
  context_length: number,
  tools: boolean,
  vision: boolean,
  json: boolean,
): ModelRow {
  return {
    id: `openrouter/${bare}`,
    bare,
    provider: 'openrouter',
    display_name,
    rating_milli,
    context_length,
    tools,
    vision,
    json,
    free: true,
  };
}

/**
 * The sealed fourteen, in the order the brief lists them. Ratings, capabilities and windows are
 * the kit's own numbers; the scorer rejects a ratings file that disagrees with them.
 */
export const SEALED_MODELS: readonly ModelRow[] = [
  row('baidu/ernie-5.5-flash-preview:free', 'Baidu: ERNIE-5.5-Flash-Preview (free)', 1150, 1024000, true, false, true),
  row('cohere/north-mini-code:free', 'Cohere: North Mini Code (free)', 1200, 256000, true, false, false),
  row('dots-studio/dots-3-note-preview:free', 'Dots Studio: Dots3-Note Preview (free)', 1150, 512000, true, true, true),
  row('google/gemma-4-26b-a4b-it:free', 'Google: Gemma 4 26B A4B (free)', 1150, 262144, true, true, true),
  row('google/gemma-4-31b-it:free', 'Google: Gemma 4 31B (free)', 1210, 262144, true, true, true),
  row('inclusionai/ling-3.2-flash:free', 'inclusionAI: Ling 3.2 Flash (free)', 1150, 1048576, true, true, true),
  row('liquid/lfm-3-1.7b:free', 'LiquidAI: LFM3-1.7B (free)', 1150, 262144, true, false, true),
  row('nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', 'NVIDIA: Nemotron 3 Nano Omni (free)', 1150, 256000, true, true, false),
  row('nvidia/nemotron-3-ultra-550b-a55b:free', 'NVIDIA: Nemotron 3 Ultra (free)', 1285, 1000000, true, false, false),
  row('nvidia/nemotron-3.5-lightning:free', 'NVIDIA: Nemotron 3.5 Lightning (free)', 1221, 1000000, true, false, false),
  row('nvidia/nemotron-3.5-super-vl:free', 'NVIDIA: Nemotron 3.5 Super VL (free)', 1303, 1000000, true, true, true),
  row('particle-ai/particle-1:free', 'Particle AI: Particle-1 (free)', 1150, 1048576, true, true, true),
  row('recursive/flint-v1.0:free', 'Recursive: Flint (free)', 1150, 262144, true, false, true),
  row('thinkingmachines/inkling-2-small:free', 'Thinking Machines: Inkling 2 Small (free)', 1475, 1048576, true, true, false),
];

export const SEALED_BY_ID = new Map(SEALED_MODELS.map((m) => [m.id, m]));

/** Whether a model's capabilities meet one prompt's requirements. */
export function meetsRequirements(model: ModelRow, requires: Requirement): boolean {
  if (requires.tools === true && !model.tools) return false;
  if (requires.vision === true && !model.vision) return false;
  if (requires.json === true && !model.json) return false;
  if (requires.min_context !== undefined && model.context_length < requires.min_context) return false;
  return true;
}

/**
 * The free models that can serve `requires`, best first: rating desc, context_length desc, id
 * asc. Models in `cooled` are skipped — a model that was refused stays out for the trial.
 */
export function rankCandidates(requires: Requirement, cooled: ReadonlySet<string>): ModelRow[] {
  return SEALED_MODELS.filter((m) => m.free && !cooled.has(m.id) && meetsRequirements(m, requires)).sort(
    (a, b) =>
      b.rating_milli - a.rating_milli ||
      b.context_length - a.context_length ||
      (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
  );
}

/**
 * The catalog: one provider, the fourteen free models. Built from the sealed table alone, so
 * `--list`, the register body and both GET /v1/models answers name exactly the same set.
 */
export function buildCatalog(): CatalogReport {
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models: SEALED_MODELS.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** The first document under this contest's prefix that carries one of `keys` as an array. */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`: its prompt rows, in the order the client sends them. */
export function readBrief(path: string): Brief {
  if (path === '') return { prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { prompts, error: null };
}
