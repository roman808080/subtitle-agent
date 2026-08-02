# Local Multilingual Subtitle Agent — v5

`subtitle_agent.py` corrects source-language SRT subtitles and produces English and/or Russian output through a local OpenAI-compatible `llama.cpp` server.

## Languages

Supported source languages:

- English: `en`
- Russian: `ru`
- Hebrew: `he`
- Czech: `cs`
- Spanish: `es`
- Portuguese: `pt`

Supported targets are English and Russian. The default is:

```bash
--targets en,ru
```

## Safety model

The model can propose subtitle text only. Python owns and preserves:

- cue count;
- original cue numbers;
- timestamps;
- cue order;
- input and reference files;
- output paths.

The model receives no shell or filesystem tools. Audio/video is never required. When `--audio` is supplied, it is read only to calculate and recheck its SHA-256, size, and modification timestamp.

For a stronger operating-system guarantee, run the agent as a user without write access to the media/input directory or mount it read-only.

## Production-hardening in v5

### Fail-closed output policy

The default is:

```bash
--output-policy strict
```

Strict mode uses schema-constrained JSON only. It never silently downgrades to generic JSON or unconstrained text.

When a failure is likely related to output size or structure, the agent automatically splits only the affected batch and retries the same strict stage. Splittable failures are:

- grammar initialization;
- request/context size;
- truncated generation;
- malformed JSON;
- stage validation failure.

Network, authentication, and server-availability errors are retried normally and are never treated as reasons to split a batch or weaken constraints.

The smallest automatic batch is controlled by:

```bash
--min-batch-size 4
```

### Optional adaptive policy

For compatibility with a server that cannot reliably use constrained JSON:

```bash
--output-policy adaptive
```

Adaptive mode first performs the same strict batch splitting. Only after a failing batch reaches `--min-batch-size` may it try:

1. generic JSON-object mode;
2. unconstrained text followed by JSON extraction and strict Python validation.

This mode is less desirable for unattended processing. Strict remains the default.

### Validated-only caching

A response enters `.subtitle_agent_cache.json` only after it passes its stage-specific validator.

On a cache hit, the response is validated again. An old or invalid cache entry is evicted and regenerated. Strict and adaptive runs use different cache identities, so strict mode cannot reuse a response produced through a weak fallback.

### Strict critic validation

Critic output is no longer silently filtered. Every issue must have exactly:

```json
{
  "id": 123,
  "severity": "error",
  "problem": "One-line explanation",
  "suggested_text": "Complete replacement text"
}
```

Unexpected IDs, duplicate IDs, invalid severities, missing fields, extra fields, and empty suggestions fail validation and trigger retry/splitting. A malformed critic response can no longer be interpreted as “no issues.”

### Diagnostics and audit trail

Failed model responses are preserved by default under:

```text
OUTPUT_DIR/diagnostics/RUN_ID/
```

Each diagnostic records:

- stage;
- response mode;
- attempt number;
- classified error;
- raw model content, when available.

Disable this when transcript content must not be retained:

```bash
--no-diagnostics
```

Successful audits include every model attempt, response mode, retry, fallback, parser repair, cache hit/eviction, and batch split.

An aborted run writes a unique failure audit such as:

```text
episode.20260802T181500.123456Z.failure.audit.json
```

### Optional separate critic model

By default, editor, critic, and reviser use the same model. This is self-review, not independent verification.

A second model served by the same llama.cpp endpoint can be selected for critic stages:

```bash
--critic-model another-model-alias
```

## Requirements

- Python 3.11 or newer recommended;
- a running `llama.cpp` OpenAI-compatible server exposing `/v1/models` and `/v1/chat/completions`;
- optional `pypdf` for PDF reference documents.

Install optional PDF support:

```bash
python -m pip install -r requirements_subtitle_agent.txt
```

## Server example

```bash
llama-server \
  -m /models/Qwen3.6-27B-Q8_0.gguf \
  --alias qwen3.6-27b \
  --host 0.0.0.0 \
  --port 8080 \
  -c 262144 \
  --reasoning off \
  --jinja
```

Restrict the listening port to the trusted LAN.

The client also sends `reasoning_effort: "none"` and `enable_thinking: false` unless `--keep-thinking` is used.

## Basic command

```bash
python subtitle_agent.py episode.srt \
  --source-lang es \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1 \
  --model qwen3.6-27b \
  --output-policy strict \
  --batch-size 48 \
  --min-batch-size 4
```

## Reference transcript inputs

References are optional and may be mixed or repeated.

### General URL

```bash
--reference-url 'https://example.org/transcript'
```

The reader follows normal redirects, parses static HTML, and accepts direct TXT, PDF, and DOCX responses. It reads only the provided URL, does not crawl links, and does not execute JavaScript.

### Local file

```bash
--reference-file transcript.txt
--reference-file saved_page.html
```

Supported local formats include TXT, HTML, HTM, Markdown, DOCX, and PDF. The flexible `--reference` option accepts either a path or an HTTP(S) URL.

## Common examples

Hebrew to English and Russian:

```bash
python subtitle_agent.py episode_he.srt \
  --source-lang he \
  --targets en,ru \
  --reference-file transcript_he.html \
  --base-url http://192.168.1.50:8080/v1
```

Czech to English with a separate critic:

```bash
python subtitle_agent.py episode_cs.srt \
  --source-lang cs \
  --targets en \
  --model qwen3.6-27b \
  --critic-model critic-model \
  --base-url http://192.168.1.50:8080/v1
```

General web reference with local fallback:

```bash
python subtitle_agent.py episode_pt.srt \
  --source-lang pt \
  --reference-url 'https://example.org/published-script' \
  --reference-file backup-script.txt \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1
```

## Outputs

For `episode.srt` with a Spanish source:

```text
episode.corrected.es.srt
episode.en.srt
episode.ru.srt
episode.audit.json
.subtitle_agent_cache.json
diagnostics/RUN_ID/                 # only created after failed attempts
```

When the source is already a requested target, that target file contains the corrected source.

## Retry behavior

`--retries` means retries of the same response constraint. It does not mean “weaken the output mode on every retry.”

The strict sequence is:

```text
schema request
  -> same-mode retry for transient/repairable output failure
  -> split affected batch when appropriate
  -> repeat with schema
  -> stop at minimum batch size if still invalid
```

The adaptive sequence adds generic JSON and plain-text extraction only at minimum batch size.

## JSON repair

The parser first attempts strict JSON. It may recover an otherwise structured response containing a raw newline, tab, or other control character inside a quoted string. Every such repair is recorded in the audit. Unsafe control bytes are removed before text reaches an SRT or audit issue field.

Repair does not bypass stage validation: IDs, fields, types, cue coverage, and nonempty text are still checked.

## Testing

```bash
python -m unittest -v test_subtitle_agent.py
```

The included tests cover:

- all supported source-language detection paths;
- local and HTTP reference parsing;
- tolerant JSON parsing;
- strict critic validation;
- classified batch splitting;
- validated-only cache behavior;
- strict versus adaptive fallback policy.

Human review remains recommended for linguistic accuracy, line breaks, reading speed, names, and timing.
