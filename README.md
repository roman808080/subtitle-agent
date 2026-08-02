# Local Multilingual Subtitle Correction and Translation Agent

`subtitle_agent.py` corrects an SRT transcript in its source language and produces English and/or Russian subtitles through a local OpenAI-compatible `llama.cpp` server.

Supported source languages:

- `en` — English
- `ru` — Russian
- `he` — Hebrew
- `cs` — Czech
- `es` — Spanish
- `pt` — Portuguese

Supported output languages:

- `en` — English
- `ru` — Russian

The default is `--targets en,ru`.

## Agentic workflow

For source correction, the program runs:

1. **Editor** — corrects transcription, spelling, punctuation, names, and terminology.
2. **Critic** — compares the correction with the original cues, neighboring context, and any reference transcript.
3. **Reviser** — applies actionable critic findings.
4. **Deterministic validator** — verifies cue coverage and output structure.

The editor–critic–reviser loop then runs independently for each requested translation.

Original cue numbers and timestamps never enter an editable model field. The model receives private consecutive IDs only for matching returned text. It can return only `{id, text}` records, and the host rejects missing, duplicated, or unexpected IDs.

## Audio and video safety

Audio/video is optional and is **not processed**. When `--audio` is supplied, the script:

- opens the file only in binary read mode to calculate SHA-256;
- never sends audio or video bytes to the model;
- never invokes FFmpeg or a shell;
- never writes to, renames, converts, or deletes the media file;
- rejects any generated path that equals a protected input path;
- verifies size, modification time, and SHA-256 again after processing.

The model receives no filesystem or shell tools.

For an operating-system-level guarantee, run the agent as a user without write permission to the media directory or expose that directory through a read-only mount.

## Requirements

- Python 3.11 or newer recommended.
- A running `llama.cpp` server exposing `/v1/chat/completions` and `/v1/models`.
- Optional: `pypdf` for local or remote PDF references.

The core agent otherwise uses only Python’s standard library.

Install optional PDF support:

```bash
python -m pip install pypdf
```

## Start the model server

Example on the inference machine:

```bash
llama-server \
  -m /models/Qwen3.6-27B-Q8_0.gguf \
  --alias qwen3.6-27b \
  --host 0.0.0.0 \
  --port 8080 \
  -c 262144 \
  --jinja
```

Restrict the port to the trusted LAN with the inference machine’s firewall.

## Basic examples

### Hebrew source to corrected Hebrew, English, and Russian

```bash
python subtitle_agent.py episode_he.srt \
  --source-lang he \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1 \
  --model qwen3.6-27b
```

The corrected source is always written separately as `episode_he.corrected.he.srt`.

### Czech source to English only

```bash
python subtitle_agent.py episode_cs.srt \
  --source-lang cs \
  --targets en \
  --base-url http://192.168.1.50:8080/v1
```

### Spanish source to Russian only

```bash
python subtitle_agent.py episode_es.srt \
  --source-lang es \
  --targets ru \
  --base-url http://192.168.1.50:8080/v1
```

### Portuguese source with automatic detection

```bash
python subtitle_agent.py episode_pt.srt \
  --source-lang auto \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1
```

Automatic detection uses script detection for Hebrew and Russian and lexical heuristics for English, Czech, Spanish, and Portuguese. When Latin-language evidence is ambiguous, the program stops and asks for an explicit `--source-lang` instead of guessing silently.

## Reference transcripts and scripts

Reference material is optional and may be supplied multiple times. All extracted text is combined and indexed locally; only the most relevant excerpts are sent with each subtitle batch.

### General HTTP(S) page

```bash
python subtitle_agent.py episode.srt \
  --source-lang es \
  --reference-url 'https://example.org/transcript-page' \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1
```

The URL reader:

- follows normal HTTP redirects;
- reads only the user-provided URL and does not crawl links;
- parses static HTML into visible text;
- accepts direct plain-text, PDF, and DOCX responses;
- enforces `--reference-max-bytes`;
- never executes JavaScript.

Pages whose transcript is rendered only by JavaScript may yield little or no text. Save or export such a page as `.html` or `.txt` and use `--reference-file`.

### Local TXT fallback

```bash
python subtitle_agent.py episode.srt \
  --source-lang he \
  --reference-file transcript.txt \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1
```

### Local HTML fallback

```bash
python subtitle_agent.py episode.srt \
  --source-lang cs \
  --reference-file saved_transcript.html \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1
```

### Mixed references

```bash
python subtitle_agent.py episode.srt \
  --source-lang pt \
  --reference-url 'https://example.org/published-script' \
  --reference-file producer_corrections.txt \
  --reference-file names_and_terms.html \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1
```

The older flexible form remains supported:

```bash
--reference https://example.org/transcript
--reference transcript.txt
```

Supported local reference formats are `.txt`, `.html`, `.htm`, `.md`, `.docx`, and `.pdf`. Other text-like files are read as plain text.

## Immutable media verification

```bash
python subtitle_agent.py episode.srt \
  --source-lang es \
  --audio /media/readonly/episode.wav \
  --reference-file transcript.txt \
  --targets en,ru \
  --base-url http://192.168.1.50:8080/v1
```

The media file is fingerprinted before and after the run. Any size, timestamp, or SHA-256 change causes the command to fail.

## Outputs

For `episode.srt` with a Spanish source and both targets:

```text
episode.corrected.es.srt  # corrected Spanish source
episode.en.srt            # English translation
episode.ru.srt            # Russian translation
episode.audit.json        # changes, review findings, references, checksums, settings
.subtitle_agent_cache.json
```

When the source is already one of the requested targets, that target file contains the corrected source. For example, an English source with `--targets en,ru` produces corrected English as both `episode.corrected.en.srt` and `episode.en.srt`.

## Recommended tuning

Defaults use 48 cues per request, four neighboring cues on each side, and up to two critic/reviser rounds.

Faster run:

```bash
python subtitle_agent.py episode.srt \
  --source-lang cs \
  --targets en,ru \
  --review-rounds 1 \
  --batch-size 64 \
  --base-url http://192.168.1.50:8080/v1
```

More conservative run:

```bash
python subtitle_agent.py episode.srt \
  --source-lang he \
  --targets en,ru \
  --review-rounds 3 \
  --batch-size 32 \
  --context-cues 6 \
  --base-url http://192.168.1.50:8080/v1
```

The cache key includes prompts, model, schema, and generation settings. Re-running the same command resumes from successful calls. Use `--no-cache` for a clean run.

## Important behavior

- Cue count, original cue numbers, ordering, and timestamps are immutable.
- The script does not merge or split cues automatically.
- Reference text is evidence, not permission to replace subtitle content wholesale.
- URL parsing is static; no browser engine or JavaScript runtime is used.
- The model’s JSON is schema-constrained where supported and validated again by Python.
- Human review remains recommended for reading speed, line breaks, names, and timing.

## llama.cpp grammar-sampler compatibility

If llama.cpp logs `error initializing grammar sampler` and shows an exact repetition such as `{47,47}`, use this version of the agent. It avoids putting the exact batch length into the JSON grammar and validates the complete cue-ID set in Python instead.

For Qwen3.6, start a recent llama-server with reasoning disabled when possible:

```bash
llama-server -m Qwen3.6-27B-Q8_0.gguf -c 262144 --reasoning off --port 8080
```

The client also sends `reasoning_effort: "none"` and retains the older `enable_thinking: false` hint for compatibility. If schema-constrained JSON still fails, the agent automatically retries with generic JSON, then unconstrained output, while preserving Python-side structural validation. Keep `--retries 3` or higher for this fallback chain.
