from __future__ import annotations

import http.server
import importlib.util
import socketserver
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("subtitle_agent.py")
spec = importlib.util.spec_from_file_location("subtitle_agent", MODULE_PATH)
assert spec and spec.loader
subtitle_agent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subtitle_agent
spec.loader.exec_module(subtitle_agent)


class QuietHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/page":
            body = b"""<!doctype html><html><head><title>Transcript</title>
            <script>ignore me</script></head><body><main><h1>Episode</h1>
            <p>Visible transcript text.</p></main></body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path == "/plain":
            body = "Texto português disponível diretamente.".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/page")
            self.end_headers()
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class SubtitleAgentTests(unittest.TestCase):
    def cue(self, text: str) -> list[subtitle_agent.Cue]:
        return [
            subtitle_agent.Cue(
                id=1,
                number=1,
                start="00:00:00,000",
                end="00:00:02,000",
                text=text,
            )
        ]

    def test_supported_language_detection(self) -> None:
        examples = {
            "en": "This is the story of a person, and we know what happened to them.",
            "he": "זה הסיפור של אדם שפחד מאוד, אבל אחר כך גילינו מה באמת קרה לו.",
            "ru": "Это история человека, который боялся, но потом мы узнали, что произошло.",
            "cs": "To je příběh člověka, který se bál, ale potom jsme zjistili, co se stalo.",
            "es": "Esta es la historia de una persona que tenía miedo, pero después supimos qué pasó.",
            "pt": "Esta é a história de uma pessoa que não tinha medo, mas depois soubemos o que aconteceu.",
        }
        for expected, text in examples.items():
            with self.subTest(expected=expected):
                self.assertEqual(expected, subtitle_agent.detect_language(self.cue(text)))

    def test_local_txt_and_html_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt = root / "script.txt"
            html = root / "script.html"
            txt.write_text("Plain transcript text", encoding="utf-8")
            html.write_text(
                "<html><body><p>Visible HTML transcript</p><script>hidden</script></body></html>",
                encoding="utf-8",
            )
            txt_text, txt_path, txt_meta = subtitle_agent.read_reference(
                str(txt), timeout=5, max_bytes=100_000, expected_kind="file"
            )
            html_text, html_path, html_meta = subtitle_agent.read_reference(
                str(html), timeout=5, max_bytes=100_000, expected_kind="file"
            )
            self.assertEqual("Plain transcript text", txt_text)
            self.assertEqual(txt.resolve(), txt_path)
            self.assertEqual("text/plain", txt_meta["content_type"])
            self.assertIn("Visible HTML transcript", html_text)
            self.assertNotIn("hidden", html_text)
            self.assertEqual(html.resolve(), html_path)
            self.assertEqual("text/html", html_meta["content_type"])

    def test_general_http_references_and_redirect(self) -> None:
        with socketserver.TCPServer(("127.0.0.1", 0), QuietHandler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                page_text, final_url, mime = subtitle_agent.read_url_text(
                    base + "/redirect", timeout=5, max_bytes=100_000
                )
                plain_text, _, plain_mime = subtitle_agent.read_url_text(
                    base + "/plain", timeout=5, max_bytes=100_000
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
            self.assertTrue(final_url.endswith("/page"))
            self.assertEqual("text/html", mime)
            self.assertIn("Visible transcript text.", page_text)
            self.assertNotIn("ignore me", page_text)
            self.assertEqual("text/plain", plain_mime)
            self.assertIn("Texto português", plain_text)

    def test_lenient_model_json_with_raw_control_characters(self) -> None:
        raw = (
            '{"issues":[{"id":97,"severity":"warning",'
            '"problem":"First line\nSecond line\twith tab",'
            '"suggested_text":"Subtitle line one\nSubtitle line two\u0000"}]}'
        )
        parsed = subtitle_agent.parse_json_object(raw)
        issues = subtitle_agent.parse_issues(parsed, {97})
        self.assertEqual(1, len(issues))
        self.assertEqual("First line Second line with tab", issues[0]["problem"])
        self.assertEqual("Subtitle line one\nSubtitle line two", issues[0]["suggested_text"])

    def test_balanced_json_extraction_ignores_trailing_text(self) -> None:
        raw = 'preface {"items":[{"id":1,"text":"brace } in text"}]} trailing'
        parsed = subtitle_agent.parse_json_object(raw)
        self.assertEqual({1: "brace } in text"}, subtitle_agent.parse_items(parsed, [1]))


    def test_critic_rejects_malformed_or_unexpected_issues(self) -> None:
        with self.assertRaises(subtitle_agent.SubtitleAgentError):
            subtitle_agent.parse_issues(
                {"issues": [{"id": "1", "severity": "warning", "problem": "x", "suggested_text": "y"}]},
                {1},
            )
        with self.assertRaises(subtitle_agent.SubtitleAgentError):
            subtitle_agent.parse_issues(
                {"issues": [{"id": 2, "severity": "warning", "problem": "x", "suggested_text": "y"}]},
                {1},
            )
        with self.assertRaises(subtitle_agent.SubtitleAgentError):
            subtitle_agent.parse_issues(
                {"issues": [{"id": 1, "severity": "maybe", "problem": "x", "suggested_text": "y"}]},
                {1},
            )

    def test_batch_splitting_only_for_splittable_failures(self) -> None:
        stats = subtitle_agent.AgentStats()
        audit: list[dict[str, object]] = []

        def operation(group: list[int]) -> tuple[int, ...]:
            if len(group) > 2:
                raise subtitle_agent.LLMStageError(
                    "test", "validation", "too large", retryable=True
                )
            return tuple(group)

        results = subtitle_agent.run_with_batch_splitting(
            list(range(1, 9)),
            operation,
            min_batch_size=2,
            stats=stats,
            split_audit=audit,
            stage_family="test",
        )
        self.assertEqual([(1, 2), (3, 4), (5, 6), (7, 8)], [value for _, value in results])
        self.assertEqual(3, stats.batch_splits)
        self.assertEqual(3, len(audit))

        with self.assertRaises(subtitle_agent.LLMStageError):
            subtitle_agent.run_with_batch_splitting(
                [1, 2, 3, 4],
                lambda group: (_ for _ in ()).throw(
                    subtitle_agent.LLMStageError("test", "network", "offline", retryable=True)
                ),
                min_batch_size=1,
                stats=subtitle_agent.AgentStats(),
                split_audit=[],
                stage_family="test",
            )

    def test_only_validated_responses_are_cached_and_bad_cache_is_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = subtitle_agent.JsonCache(root / "cache.json", set())
            stats = subtitle_agent.AgentStats()
            client = subtitle_agent.LlamaCppClient(
                base_url="http://127.0.0.1:8080/v1",
                model="test-model",
                timeout=5,
                retries=1,
                cache=cache,
                stats=stats,
                output_policy="strict",
                diagnostics_dir=None,
            )
            response_one = {
                "choices": [{"message": {"content": '{"items":[{"id":1,"text":"one"}]}'}, "finish_reason": "stop"}]
            }
            response_two = {
                "choices": [{"message": {"content": '{"items":[{"id":2,"text":"two"}]}'}, "finish_reason": "stop"}]
            }
            with mock.patch.object(subtitle_agent, "http_post_json", return_value=response_one) as post:
                value = client.complete_json(
                    system="s",
                    user="u",
                    schema=subtitle_agent.items_schema([1]),
                    max_tokens=100,
                    temperature=0.0,
                    stage="cache-test",
                    validator=lambda payload: subtitle_agent.parse_items(payload, [1]),
                    validator_name="parse_items",
                )
                self.assertEqual({1: "one"}, value)
                self.assertEqual(1, post.call_count)
                self.assertTrue(cache.data)

            # Same request identity, but stronger/different stage validation. The
            # cached response must be evicted before a fresh model call is accepted.
            with mock.patch.object(subtitle_agent, "http_post_json", return_value=response_two) as post:
                value = client.complete_json(
                    system="s",
                    user="u",
                    schema=subtitle_agent.items_schema([1]),
                    max_tokens=100,
                    temperature=0.0,
                    stage="cache-test",
                    validator=lambda payload: subtitle_agent.parse_items(payload, [2]),
                    validator_name="parse_items",
                )
                self.assertEqual({2: "two"}, value)
                self.assertEqual(1, post.call_count)
                self.assertEqual(1, stats.cache_evictions)

            bad_cache = subtitle_agent.JsonCache(root / "bad-cache.json", set())
            bad_client = subtitle_agent.LlamaCppClient(
                base_url="http://127.0.0.1:8080/v1",
                model="test-model",
                timeout=5,
                retries=1,
                cache=bad_cache,
                stats=subtitle_agent.AgentStats(),
                output_policy="strict",
                diagnostics_dir=None,
            )
            bad_response = {
                "choices": [{"message": {"content": '{"items":[{"id":99,"text":"bad"}]}'}, "finish_reason": "stop"}]
            }
            with mock.patch.object(subtitle_agent, "http_post_json", return_value=bad_response):
                with self.assertRaises(subtitle_agent.LLMStageError):
                    bad_client.complete_json(
                        system="s",
                        user="bad",
                        schema=subtitle_agent.items_schema([1]),
                        max_tokens=100,
                        temperature=0.0,
                        stage="bad-cache-test",
                        validator=lambda payload: subtitle_agent.parse_items(payload, [1]),
                        validator_name="parse_items",
                    )
            self.assertEqual({}, bad_cache.data)

    def test_strict_policy_does_not_fallback_and_adaptive_is_limited(self) -> None:
        def make_client(policy: str) -> subtitle_agent.LlamaCppClient:
            return subtitle_agent.LlamaCppClient(
                base_url="http://127.0.0.1:8080/v1",
                model="test-model",
                timeout=5,
                retries=1,
                cache=subtitle_agent.JsonCache(None, set()),
                stats=subtitle_agent.AgentStats(),
                output_policy=policy,
                diagnostics_dir=None,
            )

        grammar_error = subtitle_agent.SubtitleAgentError(
            "HTTP 400: Failed to initialize samplers: grammar error"
        )
        strict_client = make_client("strict")
        with mock.patch.object(subtitle_agent, "http_post_json", side_effect=grammar_error) as post:
            with self.assertRaises(subtitle_agent.LLMStageError) as caught:
                strict_client.complete_json(
                    system="s", user="u", schema=subtitle_agent.issues_schema(),
                    max_tokens=100, temperature=0.0, stage="strict-test",
                    validator=lambda payload: subtitle_agent.parse_issues(payload, {1}),
                    validator_name="parse_issues", allow_weak_fallback=True,
                )
            self.assertEqual("grammar", caught.exception.kind)
            self.assertEqual(1, post.call_count)

        adaptive_client = make_client("adaptive")
        success = {
            "choices": [{"message": {"content": '{"issues":[]}'}, "finish_reason": "stop"}]
        }
        payloads: list[dict[str, object]] = []

        def adaptive_side_effect(url: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
            payloads.append(payload)
            if len(payloads) == 1:
                raise grammar_error
            return success

        with mock.patch.object(subtitle_agent, "http_post_json", side_effect=adaptive_side_effect):
            value = adaptive_client.complete_json(
                system="s", user="u", schema=subtitle_agent.issues_schema(),
                max_tokens=100, temperature=0.0, stage="adaptive-test",
                validator=lambda payload: subtitle_agent.parse_issues(payload, {1}),
                validator_name="parse_issues", allow_weak_fallback=True,
            )
        self.assertEqual([], value)
        self.assertEqual("json_schema", payloads[0]["response_format"]["type"])
        self.assertEqual("json_object", payloads[1]["response_format"]["type"])

        # Availability failures never trigger weaker output constraints.
        network_client = make_client("adaptive")
        with mock.patch.object(
            subtitle_agent,
            "http_post_json",
            side_effect=subtitle_agent.SubtitleAgentError("Could not reach server: timeout"),
        ) as post:
            with self.assertRaises(subtitle_agent.LLMStageError) as caught:
                network_client.complete_json(
                    system="s", user="u", schema=subtitle_agent.issues_schema(),
                    max_tokens=100, temperature=0.0, stage="network-test",
                    validator=lambda payload: subtitle_agent.parse_issues(payload, {1}),
                    validator_name="parse_issues", allow_weak_fallback=True,
                )
            self.assertEqual("network", caught.exception.kind)
            self.assertEqual(1, post.call_count)

    def test_targets_remain_english_and_russian(self) -> None:
        self.assertEqual(["en", "ru"], subtitle_agent.parse_targets("en,ru"))
        with self.assertRaises(Exception):
            subtitle_agent.parse_targets("es")


if __name__ == "__main__":
    unittest.main()
