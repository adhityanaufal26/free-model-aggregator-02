# free-model-aggregator-02

make.fun entry: one server that aggregates the fourteen free OpenRouter models behind
the OpenAI (`POST /v1/chat/completions`) and Anthropic (`POST /v1/messages`) wires.

## Run

```sh
node main.ts            # registers with $MAKEFUN_ENV_URL, serves, exits 0 when /work says done
node main.ts --list     # prints the catalog JSON and exits 0
```

Node 22 standard library only; no install step. The gateway is `$OPENAI_BASE_URL`
(`OPENAI_API_KEY` is the trial's dummy key — the gateway swaps it for the entrant's pushed
OpenRouter key).

## Routing

Candidates are the sealed free models meeting the prompt's requirements (JSON mode, tools,
vision), ranked by rating desc, then context_length desc, then id asc. The server dials the
best candidate through the gateway's Chat Completions route, verifies the answer body carries
the prompt's token (read off the request body — the brief's prompt rows carry no token), and
retries the next-best candidate on a token miss, up to 4 attempts a prompt. A 429 / 5xx /
network error cools that model down for the rest of the trial; any other 4xx is lane-wide and
ends the prompt with a 502. Every refusal is reported on the response header
`x-makefun-upstream-failures` (at most 8 rows, in refusal order).

Best per prompt from the sealed table: non-JSON prompts → `thinkingmachines/inkling-2-small`
(1475); JSON-mode prompts → `nvidia/nemotron-3.5-super-vl` (1303); tools prompts →
`thinkingmachines/inkling-2-small` (1475). One successful call per served prompt = 6 calls, the
floor; a refused call costs nothing on the tie-break.

## Files

- `main.ts` — the server: register, poll, routes, routing policy.
- `lib/providers.ts` — the sealed model table, the brief reader, the catalog, the ranking.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation, SSE writers, the model field.
