#!/usr/bin/env python3
"""Serve a persistent Supertonic engine at POST /v1/speech."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supertonic_sima.config import add_engine_arguments, create_engine  # noqa: E402
from supertonic_sima.server import SpeechApplication, create_server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup-text", default="The speech service is ready.")
    add_engine_arguments(parser)
    args = parser.parse_args()

    with create_engine(args) as engine:
        if args.warmup_text:
            engine.synthesize(
                args.warmup_text, voice="M1", language="en", speed=1.0, seed=1101
            )
        server = create_server(args.host, args.port, SpeechApplication(engine))
        server.daemon_threads = True
        print(f"listening=http://{args.host}:{args.port}/v1/speech", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
