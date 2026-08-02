#!/usr/bin/env python3
"""
Read-only multilingual subtitle correction and EN/RU translation agent for a local llama.cpp server.

Security model
--------------
* The LLM never receives filesystem or shell tools.
* Subtitle timestamps and cue IDs are immutable.
* Audio is never sent to the model, decoded, converted, renamed, or opened for writing.
* When --audio is supplied, the file is opened only as "rb" for SHA-256 verification.
* All generated files are written atomically and may not overwrite any protected input.

The agentic loop is:
    editor -> critic -> reviser -> deterministic validator
for source correction, followed by the same loop for each requested translation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import io
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)
WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "he": "Hebrew",
    "cs": "Czech",
    "es": "Spanish",
    "pt": "Portuguese",
}
SOURCE_LANGUAGE_CODES = tuple(LANGUAGE_NAMES)
TARGET_LANGUAGE_CODES = ("en", "ru")

LATIN_LANGUAGE_MARKERS: dict[str, dict[str, set[str] | str]] = {
    "en": {
        "words": {
            "the", "and", "you", "to", "of", "is", "it", "in", "that", "we",
            "for", "are", "not", "this", "with", "was", "be", "have", "he", "she",
            "they", "but", "what", "my", "your", "do", "did", "can", "will",
        },
        "chars": "",
    },
    "cs": {
        "words": {
            "a", "že", "se", "je", "na", "to", "jsem", "jsi", "jsme", "jsou",
            "do", "pro", "ale", "co", "si", "ne", "ano", "jak", "když", "tak",
            "ten", "ta", "ty", "by", "být", "už", "mám", "má", "může", "protože",
        },
        "chars": "čďěňřšťůž",
    },
    "es": {
        "words": {
            "el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es",
            "no", "por", "para", "con", "como", "pero", "yo", "tú", "usted", "está",
            "son", "del", "al", "qué", "porque", "muy", "más", "lo", "mi", "su",
        },
        "chars": "ñ¿¡",
    },
    "pt": {
        "words": {
            "o", "a", "os", "as", "de", "que", "e", "em", "um", "uma", "é", "não",
            "por", "para", "com", "como", "mas", "eu", "você", "está", "são", "do",
            "da", "dos", "das", "porque", "muito", "mais", "meu", "minha", "seu", "sua",
        },
        "chars": "ãõçâêôà",
    },
}


class SubtitleAgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Cue:
    # id is a private, consecutive model-facing key. number is the original
    # SRT cue number and is preserved verbatim in generated files.
    id: int
    number: int
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size: int
    mtime_ns: int
    sha256: str


@dataclass
class AgentStats:
    llm_calls: int = 0
    cache_hits: int = 0
    correction_changes: int = 0
    correction_issues: int = 0
    translation_issues: dict[str, int] = dataclasses.field(default_factory=dict)


class VisibleTextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor for published transcript pages."""

    BLOCK_TAGS = {
        "p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3",
        "h4", "h5", "h6", "section", "article", "main", "blockquote", "pre",
        "ul", "ol", "table", "header", "footer", "title",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "canvas", "iframe"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    def get_text(self) -> str:
        raw = html.unescape("".join(self.parts))
        lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


class JsonCache:
    def __init__(self, path: Path | None, protected: set[Path]) -> None:
        self.path = path
        self.protected = protected
        self.data: dict[str, Any] = {}
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def get(self, key: str) -> Any | None:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        if not self.path:
            return
        self.data[key] = value
        safe_atomic_write_text(
            self.path,
            json.dumps(self.data, ensure_ascii=False, indent=2),
            self.protected,
        )


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int,
        retries: int,
        cache: JsonCache,
        stats: AgentStats,
        disable_thinking: bool = True,
    ) -> None:
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.cache = cache
        self.stats = stats
        self.disable_thinking = disable_thinking

    def resolve_model(self) -> str:
        if self.model != "auto":
            return self.model
        payload = http_get_json(f"{self.base_url}/models", timeout=self.timeout)
        models = payload.get("data") if isinstance(payload, dict) else None
        if not models or not isinstance(models, list) or not models[0].get("id"):
            raise SubtitleAgentError("Could not discover a model from /v1/models")
        self.model = str(models[0]["id"])
        return self.model

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
        stage: str,
    ) -> dict[str, Any]:
        model = self.resolve_model()
        request_identity = {
            "base_url": self.base_url,
            "model": model,
            "system": system,
            "user": user,
            "schema": schema,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stage": stage,
        }
        key = hashlib.sha256(
            json.dumps(request_identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            self.stats.cache_hits += 1
            return cached

        base_payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.disable_thinking:
            base_payload["chat_template_kwargs"] = {"enable_thinking": False}
            base_payload["reasoning_format"] = "none"

        formats = [
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "subtitle_agent_result",
                    "strict": True,
                    "schema": schema,
                },
            },
            {"type": "json_object", "schema": schema},
            {"type": "json_object"},
        ]
        last_error: Exception | None = None
        for attempt in range(self.retries):
            response_format = formats[min(attempt, len(formats) - 1)]
            payload = dict(base_payload)
            payload["response_format"] = response_format
            if attempt:
                payload["messages"] = [
                    payload["messages"][0],
                    {
                        "role": "user",
                        "content": user
                        + "\n\nYour previous response could not be parsed. Return only a JSON object "
                          "that exactly follows the requested structure; no markdown or commentary.",
                    },
                ]
            try:
                self.stats.llm_calls += 1
                raw = http_post_json(
                    f"{self.base_url}/chat/completions",
                    payload,
                    timeout=self.timeout,
                )
                content = extract_message_content(raw)
                parsed = parse_json_object(content)
                self.cache.put(key, parsed)
                return parsed
            except Exception as exc:  # retry malformed output and transient HTTP failures
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 5))
        raise SubtitleAgentError(f"LLM stage {stage!r} failed: {last_error}")


def http_post_json(url: str, payload: Mapping[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer no-key",
            "User-Agent": "subtitle-agent/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:2000]
        raise SubtitleAgentError(f"HTTP {exc.code} from {url}: {details}") from exc
    except urllib.error.URLError as exc:
        raise SubtitleAgentError(f"Could not reach {url}: {exc}") from exc
    try:
        value = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SubtitleAgentError(f"Server returned invalid JSON: {data[:500]!r}") from exc
    if not isinstance(value, dict):
        raise SubtitleAgentError("Server response was not a JSON object")
    return value


def http_get_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "User-Agent": "subtitle-agent/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise SubtitleAgentError(f"Could not GET {url}: {exc}") from exc
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise SubtitleAgentError("Server response was not a JSON object")
    return value


def extract_message_content(payload: Mapping[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SubtitleAgentError(f"Unexpected chat-completions response: {payload}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        if chunks:
            return "".join(chunks)
    raise SubtitleAgentError("The model response did not contain textual content")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = THINK_RE.sub("", text).strip()
    cleaned = FENCE_RE.sub("", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise SubtitleAgentError(f"No JSON object found in model output: {text[:1000]!r}")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise SubtitleAgentError("The model output must be one JSON object")
    return value


def normalize_timestamp(value: str) -> str:
    value = value.replace(".", ",")
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(millis):03d}"


def timestamp_ms(value: str) -> int:
    hours, minutes, rest = value.replace(".", ",").split(":")
    seconds, millis = rest.split(",")
    return (((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000) + int(millis)


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig", errors="strict")
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = re.split(r"\n{2,}", normalized)
    cues: list[Cue] = []
    for ordinal, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        try:
            cue_id = int(lines[0].strip())
            timestamp_line = lines[1].strip()
            text_lines = lines[2:]
        except ValueError:
            # Some malformed SRTs omit cue numbers. Assign a stable number but never
            # modify timestamps. This is reported in the audit.
            cue_id = ordinal
            timestamp_line = lines[0].strip()
            text_lines = lines[1:]
        match = TIMESTAMP_RE.match(timestamp_line)
        if not match:
            raise SubtitleAgentError(
                f"Invalid timestamp block near cue {cue_id}: {timestamp_line!r}"
            )
        text = "\n".join(text_lines).strip()
        cues.append(
            Cue(
                id=ordinal,
                number=cue_id,
                start=normalize_timestamp(match.group("start")),
                end=normalize_timestamp(match.group("end")),
                text=text,
            )
        )
    validate_source_cues(cues)
    return cues


def validate_source_cues(cues: Sequence[Cue]) -> None:
    if not cues:
        raise SubtitleAgentError("No subtitle cues were found")
    previous_start = -1
    for cue in cues:
        start = timestamp_ms(cue.start)
        end = timestamp_ms(cue.end)
        if start >= end:
            raise SubtitleAgentError(f"Cue {cue.id} has a non-positive duration")
        if start < previous_start:
            raise SubtitleAgentError(f"Cue {cue.id} is out of chronological order")
        previous_start = start


def compose_srt(cues: Sequence[Cue], texts: Mapping[int, str], max_line_chars: int) -> str:
    blocks: list[str] = []
    for cue in cues:
        if cue.id not in texts:
            raise SubtitleAgentError(f"Missing text for cue {cue.id}")
        wrapped = wrap_subtitle(texts[cue.id], max_line_chars=max_line_chars)
        blocks.append(f"{cue.number}\n{cue.start} --> {cue.end}\n{wrapped}")
    return "\n\n".join(blocks) + "\n"


def wrap_subtitle(text: str, max_line_chars: int) -> str:
    lines: list[str] = []
    for source_line in text.replace("\r", "").split("\n"):
        compact = re.sub(r"[ \t]+", " ", source_line).strip()
        if not compact:
            continue
        if max_line_chars <= 0 or len(compact) <= max_line_chars:
            lines.append(compact)
            continue
        lines.extend(
            textwrap.wrap(
                compact,
                width=max_line_chars,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
        )
    return "\n".join(lines).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:  # deliberately read-only
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> FileFingerprint:
    stat_result = path.stat()
    return FileFingerprint(
        path=str(path),
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        sha256=sha256_file(path),
    )


def verify_unchanged(before: FileFingerprint, path: Path) -> FileFingerprint:
    after = fingerprint(path)
    if before != after:
        raise SubtitleAgentError(
            "Protected audio changed while the agent was running. "
            f"Before={before}; after={after}"
        )
    return after


def safe_atomic_write_text(path: Path, text: str, protected: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in protected:
        raise SubtitleAgentError(f"Refusing to overwrite protected input: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=str(resolved.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, resolved)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _decode_text_bytes(data: bytes, charset: str | None = None) -> str:
    candidates = [charset, "utf-8-sig", "utf-8", "windows-1252", "latin-1"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return data.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _html_to_text(raw: str) -> str:
    parser = VisibleTextExtractor()
    parser.feed(raw)
    parser.close()
    return parser.get_text()



def _looks_like_docx(data: bytes) -> bool:
    if not data.startswith(b"PK\x03\x04"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return "word/document.xml" in archive.namelist()
    except zipfile.BadZipFile:
        return False

def read_docx_bytes(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise SubtitleAgentError("Reference is not a readable DOCX document") from exc
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return html.unescape(xml)


def read_docx(path: Path) -> str:
    return read_docx_bytes(path.read_bytes())


def read_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise SubtitleAgentError(
            "PDF reference support requires: python -m pip install pypdf"
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise SubtitleAgentError(f"Could not parse PDF reference: {exc}") from exc


def read_pdf(path: Path) -> str:
    return read_pdf_bytes(path.read_bytes())


def read_url_text(url: str, timeout: int, max_bytes: int) -> tuple[str, str, str]:
    """Read a user-provided HTTP(S) reference without crawling linked pages.

    Returns extracted text, the final URL after redirects, and the detected MIME type.
    Static HTML, plain text, direct PDF, and direct DOCX responses are supported.
    JavaScript is never executed.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SubtitleAgentError(f"Reference URL must be an absolute HTTP(S) URL: {url}")
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "subtitle-agent/2.0",
            "Accept": (
                "text/html,application/xhtml+xml,text/plain,application/pdf,"
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*;q=0.2"
            ),
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
            content_type = (response.headers.get_content_type() or "").lower()
            charset = response.headers.get_content_charset()
            final_url = response.geturl()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise SubtitleAgentError(f"Could not read reference URL {url}: {exc}") from exc
    if len(data) > max_bytes:
        raise SubtitleAgentError(f"Reference URL exceeded {max_bytes} bytes")

    final_path = urllib.parse.urlparse(final_url).path.lower()
    if content_type == "application/pdf" or final_path.endswith(".pdf") or data.startswith(b"%PDF-"):
        return read_pdf_bytes(data), final_url, "application/pdf"
    if (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or final_path.endswith(".docx")
        or _looks_like_docx(data)
    ):
        return read_docx_bytes(data), final_url, (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    decoded = _decode_text_bytes(data, charset)
    leading = decoded[:2000].lstrip().lower()
    is_html = (
        content_type in {"text/html", "application/xhtml+xml"}
        or final_path.endswith((".html", ".htm"))
        or leading.startswith("<!doctype html")
        or "<html" in leading
    )
    if is_html:
        return _html_to_text(decoded), final_url, content_type or "text/html"

    if content_type.startswith("text/") or content_type in {
        "application/json", "application/xml", "application/xhtml+xml", ""
    }:
        if "\x00" in decoded[:4000]:
            raise SubtitleAgentError(
                f"Reference URL returned unsupported binary content ({content_type or 'unknown MIME type'})"
            )
        return decoded, final_url, content_type or "text/plain"

    # Some servers mislabel downloadable text as application/octet-stream.
    if "\x00" not in decoded[:4000] and sum(ch.isprintable() or ch in "\r\n\t" for ch in decoded[:4000]) >= max(1, int(len(decoded[:4000]) * 0.85)):
        return decoded, final_url, content_type or "text/plain"
    raise SubtitleAgentError(
        f"Reference URL returned unsupported content type: {content_type or 'unknown'}"
    )


def read_reference(
    source: str,
    timeout: int,
    max_bytes: int,
    expected_kind: str | None = None,
) -> tuple[str, Path | None, dict[str, Any]]:
    parsed = urllib.parse.urlparse(source)
    is_url = parsed.scheme in {"http", "https"}
    if expected_kind == "url" and not is_url:
        raise SubtitleAgentError(f"--reference-url requires an HTTP(S) URL: {source}")
    if expected_kind == "file" and is_url:
        raise SubtitleAgentError(f"--reference-file requires a local path: {source}")

    if is_url:
        text, final_url, content_type = read_url_text(
            source, timeout=timeout, max_bytes=max_bytes
        )
        return text, None, {
            "kind": "url",
            "requested": source,
            "resolved": final_url,
            "content_type": content_type,
        }

    path = Path(source).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise SubtitleAgentError(f"Reference does not exist: {path}")
    if path.stat().st_size > max_bytes:
        raise SubtitleAgentError(f"Reference file exceeded {max_bytes} bytes: {path}")
    suffix = path.suffix.lower()
    if suffix == ".docx":
        text = read_docx(path)
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif suffix == ".pdf":
        text = read_pdf(path)
        content_type = "application/pdf"
    else:
        raw = _decode_text_bytes(path.read_bytes())
        if suffix in {".html", ".htm"} or "<html" in raw[:2000].lower():
            text = _html_to_text(raw)
            content_type = "text/html"
        else:
            text = raw
            content_type = "text/plain"
    return text, path, {
        "kind": "file",
        "requested": source,
        "resolved": str(path),
        "content_type": content_type,
    }

def clean_reference(text: str, max_chars: int) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


class ReferenceIndex:
    def __init__(self, text: str) -> None:
        paragraphs = [p.strip() for p in re.split(r"\n{1,2}", text) if p.strip()]
        # Split giant paragraphs into retrieval-size segments.
        self.paragraphs: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= 1200:
                self.paragraphs.append(paragraph)
            else:
                self.paragraphs.extend(
                    paragraph[i : i + 1200] for i in range(0, len(paragraph), 1200)
                )
        self.token_counts = [Counter(tokenize(p)) for p in self.paragraphs]
        document_frequency: Counter[str] = Counter()
        for counts in self.token_counts:
            document_frequency.update(counts.keys())
        total = max(len(self.paragraphs), 1)
        self.idf = {
            token: 1.0 + (total / (1 + freq))
            for token, freq in document_frequency.items()
        }

    def retrieve(self, query: str, max_paragraphs: int, max_chars: int) -> str:
        if not self.paragraphs:
            return ""
        query_counts = Counter(tokenize(query))
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self.token_counts):
            overlap = set(query_counts) & set(counts)
            score = sum(
                min(query_counts[token], counts[token]) * self.idf.get(token, 1.0)
                for token in overlap
            )
            # Proper nouns and longer tokens are useful for transcript matching.
            score += sum(0.5 for token in overlap if len(token) >= 8)
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)
        chosen: list[str] = []
        current = 0
        for _, index in scored[: max_paragraphs * 3]:
            paragraph = self.paragraphs[index]
            if current + len(paragraph) > max_chars:
                continue
            chosen.append(paragraph)
            current += len(paragraph)
            if len(chosen) >= max_paragraphs:
                break
        return "\n".join(chosen)


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text) if len(match.group(0)) >= 3]


def detect_language(cues: Sequence[Cue]) -> str:
    """Detect one of the supported source languages using deterministic heuristics.

    Hebrew and Russian are identified by script. English, Czech, Spanish, and
    Portuguese are scored with frequent function words and distinctive letters.
    For ambiguous Latin text, the user is asked to select --source-lang explicitly
    rather than silently choosing the wrong language.
    """
    sample = " ".join(cue.text for cue in cues[: min(300, len(cues))])
    sample = unicodedata.normalize("NFKC", sample)
    hebrew = len(HEBREW_RE.findall(sample))
    cyrillic = len(CYRILLIC_RE.findall(sample))
    latin = len(LATIN_RE.findall(sample))
    visible_letters = max(hebrew + cyrillic + latin, 1)

    if hebrew >= 8 and hebrew / visible_letters >= 0.35:
        return "he"
    if cyrillic >= 8 and cyrillic / visible_letters >= 0.35:
        return "ru"
    if latin < 8:
        raise SubtitleAgentError(
            "Could not auto-detect source language; use --source-lang "
            + ", ".join(SOURCE_LANGUAGE_CODES)
        )

    folded = sample.casefold()
    words = [match.group(0).casefold() for match in WORD_RE.finditer(folded)]
    counts = Counter(words)
    scores: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for code, markers in LATIN_LANGUAGE_MARKERS.items():
        marker_words = markers["words"]
        assert isinstance(marker_words, set)
        word_hits = sum(counts[word] for word in marker_words)
        distinctive = str(markers["chars"])
        char_hits = sum(folded.count(char) for char in distinctive)
        # Distinctive letters carry extra weight; function-word counts keep
        # unaccented subtitles detectable.
        score = float(word_hits) + 4.0 * char_hits
        scores[code] = score
        details[code] = {"word_hits": float(word_hits), "char_hits": float(char_hits)}

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_code, best_score = ranked[0]
    second_score = ranked[1][1]
    minimum = max(3.0, len(words) * 0.008)
    decisive = best_score >= minimum and (
        best_score >= second_score * 1.18 or best_score - second_score >= 4.0
    )
    if decisive:
        return best_code

    score_text = ", ".join(f"{code}={score:.1f}" for code, score in ranked)
    raise SubtitleAgentError(
        "Source language is ambiguous among Latin-script languages "
        f"({score_text}). Use --source-lang en, cs, es, or pt."
    )

def batched(values: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def batch_context(
    cues: Sequence[Cue],
    texts: Mapping[int, str],
    ids: Sequence[int],
    context_cues: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    first = ids[0]
    last = ids[-1]
    before_ids = range(max(1, first - context_cues), first)
    after_ids = range(last + 1, min(len(cues), last + context_cues) + 1)

    def records(selected: Iterable[int]) -> list[dict[str, Any]]:
        return [{"id": i, "text": texts[i]} for i in selected]

    return records(before_ids), records(ids), records(after_ids)


def items_schema(ids: Sequence[int]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def issues_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "severity": {"type": "string", "enum": ["error", "warning"]},
                        "problem": {"type": "string"},
                        "suggested_text": {"type": "string"},
                    },
                    "required": ["id", "severity", "problem", "suggested_text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["issues"],
        "additionalProperties": False,
    }


def parse_items(result: Mapping[str, Any], expected_ids: Sequence[int]) -> dict[int, str]:
    raw_items = result.get("items")
    if not isinstance(raw_items, list):
        raise SubtitleAgentError("Model response is missing an items array")
    parsed: dict[int, str] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            cue_id = int(item["id"])
            text = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if cue_id in parsed:
            raise SubtitleAgentError(f"Model returned duplicate cue ID {cue_id}")
        parsed[cue_id] = text
    expected = set(expected_ids)
    actual = set(parsed)
    if expected != actual:
        raise SubtitleAgentError(
            f"Model returned wrong cue IDs; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return parsed


def parse_issues(result: Mapping[str, Any], allowed_ids: set[int]) -> list[dict[str, Any]]:
    raw_issues = result.get("issues")
    if not isinstance(raw_issues, list):
        raise SubtitleAgentError("Model response is missing an issues array")
    issues: list[dict[str, Any]] = []
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        try:
            cue_id = int(issue["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if cue_id not in allowed_ids:
            continue
        suggested = str(issue.get("suggested_text", "")).strip()
        problem = str(issue.get("problem", "")).strip()
        severity = str(issue.get("severity", "warning")).strip().lower()
        if suggested:
            issues.append(
                {
                    "id": cue_id,
                    "severity": severity if severity in {"error", "warning"} else "warning",
                    "problem": problem,
                    "suggested_text": suggested,
                }
            )
    return issues


def correction_editor_prompt(
    source_lang: str,
    before: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    after: list[dict[str, Any]],
    reference: str,
) -> tuple[str, str]:
    language = LANGUAGE_NAMES[source_lang]
    system = f"""You are a meticulous {language} subtitle transcript editor.
You have no filesystem, shell, audio, timing, or file-writing capabilities.
Return exactly one corrected text item for every requested cue ID.
Do not translate. Preserve meaning, register, hesitation, profanity, names, and technical terms.
Correct transcription errors, spelling, punctuation, casing, and clearly broken grammar.
Use the reference excerpt only when it actually supports a correction. Never invent missing facts.
Adjacent cues may form one sentence, but every cue ID must remain present and in the same order.
Do not add speaker labels unless they already exist. Do not discuss your reasoning."""
    payload = {
        "task": "Correct requested subtitle cues in the source language.",
        "source_language": language,
        "context_before_read_only": before,
        "requested_cues": batch,
        "context_after_read_only": after,
        "reference_excerpt_may_be_empty": reference,
        "output": {"items": [{"id": "integer", "text": "corrected source-language text"}]},
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def correction_critic_prompt(
    source_lang: str,
    original: list[dict[str, Any]],
    current: list[dict[str, Any]],
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    reference: str,
) -> tuple[str, str]:
    language = LANGUAGE_NAMES[source_lang]
    system = f"""You are the independent critic for a {language} subtitle correction agent.
Find only actionable errors. Check the current version against the original, neighboring context,
and any supporting reference excerpt. Flag mistranscriptions left unfixed, unsupported rewrites,
wrong names or terminology, lost meaning, accidental translation, and broken cross-cue continuity.
Do not demand stylistic rewrites when the current text is already faithful.
For each issue, provide a complete replacement text for that one cue. Return an empty issues array
when no actionable error remains. Do not discuss timestamps or cue merging."""
    payload = {
        "task": "Critique current corrected cues.",
        "source_language": language,
        "context_before": before,
        "original_requested_cues": original,
        "current_requested_cues": current,
        "context_after": after,
        "reference_excerpt_may_be_empty": reference,
        "output": {
            "issues": [
                {
                    "id": "integer",
                    "severity": "error or warning",
                    "problem": "brief explanation",
                    "suggested_text": "complete replacement cue text",
                }
            ]
        },
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def revision_prompt(
    language: str,
    current: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    mode: str,
    source: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    system = f"""You are the reviser in a subtitle {mode} agent.
Apply the critic's valid fixes conservatively. Return exactly one item for every requested cue ID,
including unchanged cues. Preserve IDs and meaning. The output language must be {language}.
Do not merge, omit, renumber, or add cues. Do not explain your work."""
    payload: dict[str, Any] = {
        "task": f"Revise the current {mode} using the critic issues.",
        "required_output_language": language,
        "context_before": before,
        "current_requested_cues": current,
        "critic_issues": issues,
        "context_after": after,
        "output": {"items": [{"id": "integer", "text": "final cue text"}]},
    }
    if source is not None:
        payload["source_cues"] = source
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def translation_editor_prompt(
    source_lang: str,
    target_lang: str,
    before_source: list[dict[str, Any]],
    source_batch: list[dict[str, Any]],
    after_source: list[dict[str, Any]],
) -> tuple[str, str]:
    source_name = LANGUAGE_NAMES[source_lang]
    target_name = LANGUAGE_NAMES[target_lang]
    system = f"""You are a professional subtitle translator from {source_name} to {target_name}.
You have no filesystem, shell, audio, timing, or file-writing capabilities.
Return exactly one translated text item for every requested cue ID.
Translate meaning and tone naturally, using neighboring cues to resolve sentence context.
Preserve names, numbers, technical terms, emphasis tags, and intentional sound descriptions.
Do not summarize, censor, add information, merge cues, or leave source-language text untranslated
unless it is a proper noun or established term. Do not explain your work."""
    payload = {
        "task": f"Translate requested cues from {source_name} to {target_name}.",
        "context_before_source_read_only": before_source,
        "requested_source_cues": source_batch,
        "context_after_source_read_only": after_source,
        "output": {"items": [{"id": "integer", "text": f"natural {target_name} subtitle"}]},
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def translation_critic_prompt(
    source_lang: str,
    target_lang: str,
    source_batch: list[dict[str, Any]],
    current_target: list[dict[str, Any]],
    before_target: list[dict[str, Any]],
    after_target: list[dict[str, Any]],
) -> tuple[str, str]:
    source_name = LANGUAGE_NAMES[source_lang]
    target_name = LANGUAGE_NAMES[target_lang]
    system = f"""You are the independent critic for {source_name}-to-{target_name} subtitle translation.
Find only actionable errors: omissions, additions, wrong meaning, wrong names or numbers, untranslated
ordinary source text, unnatural target-language grammar, inconsistent terminology, and broken
cross-cue continuity. Respect subtitle brevity, but do not shorten away meaning.
For each issue provide a complete replacement in {target_name}. Return an empty issues array when
no actionable error remains. Do not request cue merging or timestamp changes."""
    payload = {
        "task": "Critique the current translation.",
        "source_cues": source_batch,
        "current_target_cues": current_target,
        "context_before_target": before_target,
        "context_after_target": after_target,
        "output": {
            "issues": [
                {
                    "id": "integer",
                    "severity": "error or warning",
                    "problem": "brief explanation",
                    "suggested_text": f"complete replacement in {target_name}",
                }
            ]
        },
    }
    return system, json.dumps(payload, ensure_ascii=False, indent=2)


def retrieve_reference(
    index: ReferenceIndex | None,
    batch_records: Sequence[dict[str, Any]],
    max_paragraphs: int,
    max_chars: int,
) -> str:
    if not index:
        return ""
    query = " ".join(str(record["text"]) for record in batch_records)
    return index.retrieve(query, max_paragraphs=max_paragraphs, max_chars=max_chars)


def correct_source(
    *,
    client: LlamaCppClient,
    cues: Sequence[Cue],
    source_lang: str,
    reference_index: ReferenceIndex | None,
    batch_size: int,
    context_cues: int,
    review_rounds: int,
    reference_paragraphs: int,
    reference_chars: int,
    max_tokens: int,
    stats: AgentStats,
    audit: dict[str, Any],
) -> dict[int, str]:
    original = {cue.id: cue.text for cue in cues}
    current = dict(original)
    ids = [cue.id for cue in cues]
    stage_audit: list[dict[str, Any]] = []

    print(f"[1/3] Correcting {len(cues)} {LANGUAGE_NAMES[source_lang]} cues...", file=sys.stderr)
    for group in batched(ids, batch_size):
        before, batch, after = batch_context(cues, current, group, context_cues)
        reference = retrieve_reference(
            reference_index, batch, reference_paragraphs, reference_chars
        )
        system, user = correction_editor_prompt(source_lang, before, batch, after, reference)
        result = client.complete_json(
            system=system,
            user=user,
            schema=items_schema(group),
            max_tokens=max_tokens,
            temperature=0.05,
            stage=f"correction-editor:{group[0]}-{group[-1]}",
        )
        current.update(parse_items(result, group))

    for round_number in range(1, review_rounds + 1):
        round_issues = 0
        for group in batched(ids, batch_size):
            before, current_batch, after = batch_context(cues, current, group, context_cues)
            original_batch = [{"id": cue_id, "text": original[cue_id]} for cue_id in group]
            reference = retrieve_reference(
                reference_index, original_batch, reference_paragraphs, reference_chars
            )
            system, user = correction_critic_prompt(
                source_lang,
                original_batch,
                current_batch,
                before,
                after,
                reference,
            )
            critique = client.complete_json(
                system=system,
                user=user,
                schema=issues_schema(),
                max_tokens=max_tokens,
                temperature=0.0,
                stage=f"correction-critic-r{round_number}:{group[0]}-{group[-1]}",
            )
            issues = parse_issues(critique, set(group))
            if not issues:
                continue
            round_issues += len(issues)
            stats.correction_issues += len(issues)
            system, user = revision_prompt(
                LANGUAGE_NAMES[source_lang],
                current_batch,
                issues,
                before,
                after,
                mode="correction",
            )
            revised = client.complete_json(
                system=system,
                user=user,
                schema=items_schema(group),
                max_tokens=max_tokens,
                temperature=0.0,
                stage=f"correction-reviser-r{round_number}:{group[0]}-{group[-1]}",
            )
            current.update(parse_items(revised, group))
            stage_audit.append(
                {
                    "round": round_number,
                    "batch": [group[0], group[-1]],
                    "issues": issues,
                }
            )
        print(
            f"      correction review round {round_number}: {round_issues} issue(s)",
            file=sys.stderr,
        )
        if round_issues == 0:
            break

    changes = [
        {
            "id": cue_id,
            "cue_number": cues[cue_id - 1].number,
            "original": original[cue_id],
            "corrected": current[cue_id],
        }
        for cue_id in ids
        if normalize_compare(original[cue_id]) != normalize_compare(current[cue_id])
    ]
    stats.correction_changes = len(changes)
    audit["correction"] = {"changes": changes, "review_batches": stage_audit}
    return current


def translate_target(
    *,
    client: LlamaCppClient,
    cues: Sequence[Cue],
    source_texts: Mapping[int, str],
    source_lang: str,
    target_lang: str,
    batch_size: int,
    context_cues: int,
    review_rounds: int,
    max_tokens: int,
    stats: AgentStats,
    audit: dict[str, Any],
) -> dict[int, str]:
    if target_lang == source_lang:
        return dict(source_texts)

    ids = [cue.id for cue in cues]
    target: dict[int, str] = {cue.id: "" for cue in cues}
    print(
        f"[2/3] Translating {LANGUAGE_NAMES[source_lang]} -> {LANGUAGE_NAMES[target_lang]}...",
        file=sys.stderr,
    )
    for group in batched(ids, batch_size):
        before_source, source_batch, after_source = batch_context(
            cues, source_texts, group, context_cues
        )
        system, user = translation_editor_prompt(
            source_lang, target_lang, before_source, source_batch, after_source
        )
        result = client.complete_json(
            system=system,
            user=user,
            schema=items_schema(group),
            max_tokens=max_tokens,
            temperature=0.1,
            stage=f"translation-{target_lang}-editor:{group[0]}-{group[-1]}",
        )
        target.update(parse_items(result, group))

    target_audit: list[dict[str, Any]] = []
    total_issues = 0
    for round_number in range(1, review_rounds + 1):
        round_issues = 0
        for group in batched(ids, batch_size):
            _, source_batch, _ = batch_context(cues, source_texts, group, context_cues)
            before_target, target_batch, after_target = batch_context(
                cues, target, group, context_cues
            )
            system, user = translation_critic_prompt(
                source_lang,
                target_lang,
                source_batch,
                target_batch,
                before_target,
                after_target,
            )
            critique = client.complete_json(
                system=system,
                user=user,
                schema=issues_schema(),
                max_tokens=max_tokens,
                temperature=0.0,
                stage=f"translation-{target_lang}-critic-r{round_number}:{group[0]}-{group[-1]}",
            )
            issues = parse_issues(critique, set(group))
            if not issues:
                continue
            round_issues += len(issues)
            total_issues += len(issues)
            system, user = revision_prompt(
                LANGUAGE_NAMES[target_lang],
                target_batch,
                issues,
                before_target,
                after_target,
                mode="translation",
                source=source_batch,
            )
            revised = client.complete_json(
                system=system,
                user=user,
                schema=items_schema(group),
                max_tokens=max_tokens,
                temperature=0.0,
                stage=f"translation-{target_lang}-reviser-r{round_number}:{group[0]}-{group[-1]}",
            )
            target.update(parse_items(revised, group))
            target_audit.append(
                {
                    "round": round_number,
                    "batch": [group[0], group[-1]],
                    "issues": issues,
                }
            )
        print(
            f"      {target_lang} review round {round_number}: {round_issues} issue(s)",
            file=sys.stderr,
        )
        if round_issues == 0:
            break

    stats.translation_issues[target_lang] = total_issues
    audit.setdefault("translations", {})[target_lang] = {
        "review_batches": target_audit,
    }
    return target


def normalize_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def deterministic_validate(
    cues: Sequence[Cue],
    texts: Mapping[int, str],
    language: str,
) -> list[str]:
    warnings: list[str] = []
    expected = {cue.id for cue in cues}
    actual = set(texts)
    if expected != actual:
        raise SubtitleAgentError(
            f"Validation failed for {language}: missing={expected-actual}, extra={actual-expected}"
        )
    for cue in cues:
        text = texts[cue.id].strip()
        if not text:
            raise SubtitleAgentError(f"Validation failed: empty {language} cue {cue.id}")
        if "-->" in text:
            warnings.append(f"Cue {cue.id} contains a timestamp arrow in text")
        if len(text) > 500:
            warnings.append(f"Cue {cue.id} is unusually long ({len(text)} characters)")
    joined = " ".join(texts.values())
    cyr = len(CYRILLIC_RE.findall(joined))
    heb = len(HEBREW_RE.findall(joined))
    lat = len(LATIN_RE.findall(joined))
    if language == "ru" and cyr < max(10, (lat + heb) // 5):
        warnings.append("Russian output contains unexpectedly little Cyrillic text")
    if language == "he" and heb < max(10, (lat + cyr) // 5):
        warnings.append("Hebrew output contains unexpectedly little Hebrew text")
    if language in {"en", "cs", "es", "pt"} and lat < max(10, (cyr + heb) // 5):
        warnings.append(
            f"{LANGUAGE_NAMES[language]} output contains unexpectedly little Latin text"
        )
    return warnings


def parse_targets(value: str) -> list[str]:
    targets: list[str] = []
    for item in value.split(","):
        code = item.strip().lower()
        if not code:
            continue
        if code not in TARGET_LANGUAGE_CODES:
            raise argparse.ArgumentTypeError("Targets must be en, ru, or en,ru")
        if code not in targets:
            targets.append(code)
    if not targets:
        raise argparse.ArgumentTypeError("At least one target language is required")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correct and translate SRT subtitles with a local llama.cpp model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_srt", type=Path, help="Source .srt file")
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=parse_targets("en,ru"),
        help="Comma-separated output languages: en, ru, or en,ru",
    )
    parser.add_argument(
        "--source-lang",
        choices=["auto", *SOURCE_LANGUAGE_CODES],
        default="auto",
        help="Source subtitle language: auto, en, ru, he, cs, es, or pt",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help=(
            "Optional transcript/script local path or HTTP(S) URL; supports static HTML, "
            "TXT, Markdown, DOCX, and PDF; may be repeated"
        ),
    )
    parser.add_argument(
        "--reference-url",
        action="append",
        default=[],
        help="Optional user-provided HTTP(S) reference URL; may be repeated",
    )
    parser.add_argument(
        "--reference-file",
        action="append",
        default=[],
        help="Optional local reference file (.txt, .html, .md, .docx, or .pdf); may be repeated",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional audio/video file to SHA-256 verify as immutable; it is never sent to the model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subtitle_agent_output"),
        help="Directory for generated files",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080/v1",
        help="llama.cpp OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model ID/alias; auto uses the first /v1/models entry",
    )
    parser.add_argument("--batch-size", type=int, default=48, help="Cues per model batch")
    parser.add_argument("--context-cues", type=int, default=4, help="Neighboring cues per side")
    parser.add_argument(
        "--review-rounds", type=int, default=2, help="Maximum critic/reviser rounds per stage"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192, help="Maximum generated tokens per LLM call"
    )
    parser.add_argument(
        "--max-line-chars", type=int, default=44, help="Soft output wrapping width; 0 disables"
    )
    parser.add_argument("--timeout", type=int, default=900, help="HTTP timeout per LLM call, seconds")
    parser.add_argument("--retries", type=int, default=3, help="LLM request/JSON retries")
    parser.add_argument(
        "--reference-max-bytes", type=int, default=20_000_000, help="Maximum downloaded reference size"
    )
    parser.add_argument(
        "--reference-max-chars", type=int, default=2_000_000, help="Maximum combined reference text"
    )
    parser.add_argument(
        "--reference-paragraphs", type=int, default=24, help="Retrieved reference segments per batch"
    )
    parser.add_argument(
        "--reference-chars", type=int, default=18_000, help="Maximum reference excerpt per batch"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable response caching/resume support"
    )
    parser.add_argument(
        "--keep-thinking", action="store_true", help="Do not request enable_thinking=false"
    )
    return parser


def check_arguments(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise SubtitleAgentError("--batch-size must be positive")
    if args.context_cues < 0:
        raise SubtitleAgentError("--context-cues cannot be negative")
    if args.review_rounds < 0:
        raise SubtitleAgentError("--review-rounds cannot be negative")
    if args.max_tokens < 256:
        raise SubtitleAgentError("--max-tokens is too small")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        check_arguments(args)
        input_srt = args.input_srt.expanduser().resolve()
        if not input_srt.is_file():
            raise SubtitleAgentError(f"Input SRT does not exist: {input_srt}")
        output_dir = args.output_dir.expanduser().resolve()

        protected: set[Path] = {input_srt}
        audio_before: FileFingerprint | None = None
        audio_path: Path | None = None
        if args.audio:
            audio_path = args.audio.expanduser().resolve()
            if not audio_path.is_file():
                raise SubtitleAgentError(f"Audio/video file does not exist: {audio_path}")
            protected.add(audio_path)
            audio_before = fingerprint(audio_path)
            print(
                f"Protected audio fingerprint: {audio_before.sha256} ({audio_before.size} bytes)",
                file=sys.stderr,
            )

        reference_parts: list[str] = []
        reference_sources: list[dict[str, Any]] = []
        reference_inputs: list[tuple[str, str | None]] = [
            *((source, None) for source in args.reference),
            *((source, "url") for source in args.reference_url),
            *((source, "file") for source in args.reference_file),
        ]
        for source, expected_kind in reference_inputs:
            text, local_path, source_metadata = read_reference(
                source,
                timeout=args.timeout,
                max_bytes=args.reference_max_bytes,
                expected_kind=expected_kind,
            )
            if local_path:
                protected.add(local_path)
            clean = clean_reference(text, args.reference_max_chars)
            if not clean:
                raise SubtitleAgentError(f"Reference contains no extractable text: {source}")
            reference_parts.append(clean)
            reference_sources.append(
                {
                    **source_metadata,
                    "characters": len(clean),
                    "sha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
                }
            )
        combined_reference = clean_reference(
            "\n\n".join(reference_parts), args.reference_max_chars
        )
        reference_index = ReferenceIndex(combined_reference) if combined_reference else None

        cues = parse_srt(input_srt)
        source_lang = detect_language(cues) if args.source_lang == "auto" else args.source_lang
        targets: list[str] = args.targets

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = None if args.no_cache else output_dir / ".subtitle_agent_cache.json"
        cache = JsonCache(cache_path, protected)
        stats = AgentStats()
        client = LlamaCppClient(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            cache=cache,
            stats=stats,
            disable_thinking=not args.keep_thinking,
        )
        model = client.resolve_model()

        audit: dict[str, Any] = {
            "agent_version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_srt": str(input_srt),
            "input_sha256": sha256_file(input_srt),
            "cue_count": len(cues),
            "source_language": source_lang,
            "targets": targets,
            "reference_sources": reference_sources,
            "model": model,
            "base_url": args.base_url,
            "configuration": {
                "batch_size": args.batch_size,
                "context_cues": args.context_cues,
                "review_rounds": args.review_rounds,
                "max_tokens": args.max_tokens,
                "max_line_chars": args.max_line_chars,
            },
            "security": {
                "audio_sent_to_model": False,
                "audio_opened_for_write": False,
                "timestamps_mutable": False,
                "cue_ids_mutable": False,
                "llm_has_file_tools": False,
                "llm_has_shell_tools": False,
                "reference_urls_user_supplied_only": True,
                "reference_link_crawling": False,
                "reference_javascript_execution": False,
                "reference_files_opened_for_write": False,
            },
            "audio_before": dataclasses.asdict(audio_before) if audio_before else None,
        }

        corrected = correct_source(
            client=client,
            cues=cues,
            source_lang=source_lang,
            reference_index=reference_index,
            batch_size=args.batch_size,
            context_cues=args.context_cues,
            review_rounds=args.review_rounds,
            reference_paragraphs=args.reference_paragraphs,
            reference_chars=args.reference_chars,
            max_tokens=args.max_tokens,
            stats=stats,
            audit=audit,
        )

        outputs: dict[str, dict[int, str]] = {}
        for target_lang in targets:
            outputs[target_lang] = translate_target(
                client=client,
                cues=cues,
                source_texts=corrected,
                source_lang=source_lang,
                target_lang=target_lang,
                batch_size=args.batch_size,
                context_cues=args.context_cues,
                review_rounds=args.review_rounds,
                max_tokens=args.max_tokens,
                stats=stats,
                audit=audit,
            )

        print("[3/3] Validating and writing new files...", file=sys.stderr)
        stem = input_srt.stem
        generated_files: dict[str, str] = {}
        warnings: dict[str, list[str]] = {}

        corrected_path = output_dir / f"{stem}.corrected.{source_lang}.srt"
        warnings[f"corrected_{source_lang}"] = deterministic_validate(
            cues, corrected, source_lang
        )
        safe_atomic_write_text(
            corrected_path,
            compose_srt(cues, corrected, args.max_line_chars),
            protected,
        )
        generated_files[f"corrected_{source_lang}"] = str(corrected_path)

        for target_lang, texts in outputs.items():
            path = output_dir / f"{stem}.{target_lang}.srt"
            warnings[target_lang] = deterministic_validate(cues, texts, target_lang)
            safe_atomic_write_text(
                path,
                compose_srt(cues, texts, args.max_line_chars),
                protected,
            )
            generated_files[target_lang] = str(path)

        audio_after = None
        if audio_before and audio_path:
            audio_after = verify_unchanged(audio_before, audio_path)
            print("Protected audio checksum is unchanged.", file=sys.stderr)

        audit["audio_after"] = dataclasses.asdict(audio_after) if audio_after else None
        audit["generated_files"] = generated_files
        audit["warnings"] = warnings
        audit["statistics"] = dataclasses.asdict(stats)
        audit_path = output_dir / f"{stem}.audit.json"
        safe_atomic_write_text(
            audit_path,
            json.dumps(audit, ensure_ascii=False, indent=2),
            protected,
        )
        generated_files["audit"] = str(audit_path)

        print(json.dumps({"ok": True, "files": generated_files}, ensure_ascii=False, indent=2))
        return 0
    except (SubtitleAgentError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
