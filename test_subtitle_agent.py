from __future__ import annotations

import http.server
import importlib.util
import socketserver
import sys
import tempfile
import threading
import unittest
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

    def test_targets_remain_english_and_russian(self) -> None:
        self.assertEqual(["en", "ru"], subtitle_agent.parse_targets("en,ru"))
        with self.assertRaises(Exception):
            subtitle_agent.parse_targets("es")


if __name__ == "__main__":
    unittest.main()
