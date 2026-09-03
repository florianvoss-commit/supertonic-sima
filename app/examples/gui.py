#!/usr/bin/env python3
"""Serve a browser UI for trying Supertonic voices and languages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supertonic_sima.config import add_engine_arguments, create_engine  # noqa: E402
from supertonic_sima.server import SpeechApplication, create_server  # noqa: E402


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Supertonic 3 · SiMa Modalix</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10131a;color:#edf2f7}
body{max-width:760px;margin:8vh auto;padding:0 24px}h1{font-size:2rem;margin-bottom:.25rem}
p{color:#aeb8c5}form{display:grid;gap:14px;background:#191e28;padding:24px;border-radius:16px}
textarea,select,input,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #3d4757;background:#111620;color:inherit}
textarea{min-height:140px;resize:vertical}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
label{display:grid;gap:6px;font-size:.85rem;color:#bec8d5}button{background:#55c2ff;color:#07111a;font-weight:700;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}audio{width:100%;margin-top:20px}#status{min-height:1.5rem}
@media(max-width:650px){.row{grid-template-columns:1fr 1fr}}
</style></head><body><h1>Supertonic 3</h1><p>Hybrid ONNX Runtime + Modalix MLA speech synthesis.</p>
<form id="form"><label>Text<textarea id="text">Hello from the SiMa Modalix DevKit.</textarea></label>
<div class="row"><label>Voice<select id="voice"></select></label><label>Language<select id="language"></select></label>
<label>Speed<input id="speed" type="number" min="0.7" max="2" step="0.1" value="1"></label>
<label>Seed<input id="seed" type="number" value="1101"></label></div><button id="submit">Generate speech</button></form>
<p id="status"></p><audio id="audio" controls></audio>
<script>
const form=document.querySelector('#form'), status=document.querySelector('#status'), button=document.querySelector('#submit'), audio=document.querySelector('#audio');
const text=document.querySelector('#text'), voice=document.querySelector('#voice'), language=document.querySelector('#language'), speed=document.querySelector('#speed'), seed=document.querySelector('#seed');
fetch('/config').then(r=>r.json()).then(c=>{for(const v of c.voices)voice.add(new Option(v,v));voice.value='M1';for(const l of c.languages)language.add(new Option(l,l));language.value='en'});
form.onsubmit=async e=>{e.preventDefault();button.disabled=true;status.textContent='Generating…';
try{const r=await fetch('/v1/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:text.value,voice:voice.value,language:language.value,speed:Number(speed.value),seed:Number(seed.value)})});
if(!r.ok)throw new Error((await r.json()).error);const blob=await r.blob();audio.src=URL.createObjectURL(blob);audio.play();
status.textContent=`Audio ${r.headers.get('X-Audio-Length-Seconds')} s · generation ${r.headers.get('X-Generation-Length-Seconds')} s · RTF ${r.headers.get('X-Real-Time-Factor')}`;
}catch(error){status.textContent=`Error: ${error.message}`}finally{button.disabled=false}};
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--warmup-text", default="The speech demo is ready.")
    add_engine_arguments(parser)
    args = parser.parse_args()

    with create_engine(args) as engine:
        if args.warmup_text:
            engine.synthesize(
                args.warmup_text, voice="M1", language="en", speed=1.0, seed=1101
            )
        server = create_server(
            args.host, args.port, SpeechApplication(engine, index_html=INDEX_HTML)
        )
        server.daemon_threads = True
        print(f"gui=http://{args.host}:{args.port}/", flush=True)
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
