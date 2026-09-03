"""Small standard-library HTTP server around one persistent TTS engine."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock
from typing import Any

from .audio import wav_bytes
from .engine import SupertonicModalix
from .text import AVAILABLE_LANGUAGES, AVAILABLE_VOICES, MAX_SPEED, MIN_SPEED


class SpeechApplication:
    def __init__(self, engine: SupertonicModalix, index_html: str | None = None) -> None:
        self.engine = engine
        self.index_html = index_html
        self.lock = Lock()

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
        text = payload.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'input' must be a non-empty string")
        response_format = payload.get("response_format", "wav")
        if response_format != "wav":
            raise ValueError("only response_format='wav' is supported")
        voice = str(payload.get("voice", "M1"))
        language = str(payload.get("language", payload.get("lang", "en")))
        speed = float(payload.get("speed", 1.0))
        seed = int(payload.get("seed", 1101))
        if voice not in AVAILABLE_VOICES:
            raise ValueError(f"unsupported voice {voice!r}")
        if language not in AVAILABLE_LANGUAGES:
            raise ValueError(f"unsupported language {language!r}")
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        with self.lock:
            result = self.engine.synthesize(
                text, voice=voice, language=language, speed=speed, seed=seed
            )
        body = wav_bytes(result.waveform, result.sample_rate)
        headers = {
            "X-Audio-Length-Seconds": f"{result.audio_seconds:.6f}",
            "X-Generation-Length-Seconds": f"{result.generation_seconds:.6f}",
            "X-Real-Time-Factor": f"{result.real_time_factor:.6f}",
            "X-Latent-Length": str(result.latent_length),
            "X-Text-Length": str(result.text_length),
            "X-Denoising-Steps": str(self.engine.steps),
        }
        return body, headers


def create_server(
    host: str,
    port: int,
    application: SpeechApplication,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SupertonicSiMa/1"

        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(
                status,
                json.dumps(payload, sort_keys=True).encode(),
                "application/json",
            )

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "vocoder_backend": application.engine.vocoder_backend,
                        "steps": application.engine.steps,
                    },
                )
            elif self.path == "/config":
                self._json(
                    HTTPStatus.OK,
                    {
                        "languages": sorted(AVAILABLE_LANGUAGES),
                        "voices": AVAILABLE_VOICES,
                        "min_speed": MIN_SPEED,
                        "max_speed": MAX_SPEED,
                        "steps": application.engine.steps,
                    },
                )
            elif self.path == "/" and application.index_html is not None:
                self._send(
                    HTTPStatus.OK,
                    application.index_html.encode(),
                    "text/html; charset=utf-8",
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/speech":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1_000_000:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                raw_text = payload.get("input", "")
                raw_chars = len(raw_text) if isinstance(raw_text, str) else 0
                logged_text = json.dumps(raw_text, ensure_ascii=False)
                chunk_index = payload.get("chunk_index", 1)
                chunk_count = payload.get("chunk_count", 1)
                source_chars = payload.get("source_chars", raw_chars)
                boundary = str(payload.get("split_boundary", "single"))
                boundary = boundary.replace("\r", "").replace("\n", "")[:24]
                self.log_message(
                    "synthesis start chunk=%s/%s boundary=%s "
                    "raw_chars=%d source_chars=%s language=%s voice=%s "
                    "speed=%s seed=%s steps=%d text=%s",
                    chunk_index,
                    chunk_count,
                    boundary,
                    raw_chars,
                    source_chars,
                    payload.get("language", payload.get("lang", "en")),
                    payload.get("voice", "M1"),
                    payload.get("speed", 1.0),
                    payload.get("seed", 1101),
                    application.engine.steps,
                    logged_text,
                )
                body, headers = application.synthesize(payload)
                self.log_message(
                    "synthesis done chunk=%s/%s processed_chars=%s "
                    "latent_frames=%s audio_s=%s generation_s=%s rtf=%s",
                    chunk_index,
                    chunk_count,
                    headers["X-Text-Length"],
                    headers["X-Latent-Length"],
                    headers["X-Audio-Length-Seconds"],
                    headers["X-Generation-Length-Seconds"],
                    headers["X-Real-Time-Factor"],
                )
                self._send(HTTPStatus.OK, body, "audio/wav", headers)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # noqa: BLE001
                self.log_error("synthesis failed: %s", error)
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    return ThreadingHTTPServer((host, port), Handler)
