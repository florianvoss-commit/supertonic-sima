#!/usr/bin/env python3
"""Serve a browser UI for trying Supertonic voices and languages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supertonic_sima.config import add_engine_arguments, create_engine  # noqa: E402
from supertonic_sima.server import SpeechApplication, create_server  # noqa: E402


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Supertonic 3 · SiMa Modalix</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10131a;color:#edf2f7}
body{max-width:760px;margin:8vh auto;padding:0 24px}h1{font-size:2rem;margin-bottom:.25rem}
p{color:#aeb8c5}form{display:grid;gap:14px;background:#191e28;padding:24px;border-radius:16px}
textarea,select,input,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #3d4757;background:#111620;color:inherit}
textarea{min-height:140px;resize:vertical}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
label{display:grid;gap:6px;font-size:.85rem;color:#bec8d5}button{background:#55c2ff;color:#07111a;font-weight:700;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}.actions{display:grid;grid-template-columns:2fr 1fr;gap:12px}
#stop{background:#2a3341;color:#edf2f7}#status{min-height:1.5rem}
@media(max-width:650px){.row{grid-template-columns:1fr 1fr}}
</style></head><body><h1>Supertonic 3</h1><p>Hybrid ONNX Runtime + Modalix MLA speech synthesis.</p>
<form id="form"><label>Text<textarea id="text">Hello from the SiMa Modalix DevKit.</textarea></label>
<div class="row"><label>Voice<select id="voice"></select></label><label>Language<select id="language"></select></label>
<label>Speed<input id="speed" type="number" min="0.7" max="2" step="0.1" value="1"></label>
<label>Seed<input id="seed" type="number" value="1101"></label></div>
<div class="actions"><button id="submit">Generate and play</button><button id="stop" type="button">Stop</button></div></form>
<p id="status"></p>
<script>
const form=document.querySelector('#form'), status=document.querySelector('#status'), button=document.querySelector('#submit'), stopButton=document.querySelector('#stop');
const text=document.querySelector('#text'), voice=document.querySelector('#voice'), language=document.querySelector('#language'), speed=document.querySelector('#speed'), seed=document.querySelector('#seed');
fetch('/config').then(r=>r.json()).then(c=>{for(const v of c.voices)voice.add(new Option(v,v));voice.value='M1';for(const l of c.languages)language.add(new Option(l,l));language.value='en'});
const MAX_MODEL_CHARS=192;
let generation=0, controller=null, audioContext=null, activeSources=[];

function hasEndingPunctuation(value){
  return /[.!?;:,'"')\]}\u2026。」】〉》›»]$/u.test(value);
}

function fitsChunk(value,maxChars){
  return value.length+(hasEndingPunctuation(value)?0:1)<=maxChars;
}

function splitLongSegment(segment,maxChars){
  const parts=[];
  let remaining=segment.trim();
  while(!fitsChunk(remaining,maxChars)){
    const window=remaining.slice(0,maxChars+1);
    let cut=Math.max(window.lastIndexOf(','),window.lastIndexOf('，'));
    if(cut<Math.floor(maxChars*.5)){
      cut=remaining.slice(0,maxChars).lastIndexOf(' ');
      if(cut<1)cut=maxChars-1;
    }else cut+=1;
    parts.push(remaining.slice(0,cut).trim());
    remaining=remaining.slice(cut).trim();
  }
  if(remaining)parts.push(remaining);
  return parts;
}

function sentenceChunks(value,lang){
  const wrapperChars=`<${lang}></${lang}>`.length;
  const maxChars=MAX_MODEL_CHARS-wrapperChars;
  const cleaned=value.normalize('NFKD').replace(/\s+/gu,' ').trim();
  if(!cleaned)return [];
  // Cover Latin/Korean punctuation plus Japanese/CJK full-width boundaries.
  // Keep closing quotes/brackets attached to the sentence they terminate.
  const boundaries='.!?:;…。！？：；｡';
  const closers='"\u0027”’)\\]」』】〉》›»';
  const pattern=new RegExp(`[^${boundaries}]+[${boundaries}]+[${closers}]*|[^${boundaries}]+$`,'gu');
  const sentences=(cleaned.match(pattern)||[cleaned]).map(item=>item.trim()).filter(Boolean);
  const chunks=[];
  let current='';
  for(const sentence of sentences){
    for(const part of splitLongSegment(sentence,maxChars)){
      const candidate=current?`${current} ${part}`:part;
      if(fitsChunk(candidate,maxChars)){current=candidate;continue;}
      if(current)chunks.push(current);
      current=part;
    }
  }
  if(current)chunks.push(current);
  return chunks;
}

function splitBoundary(value,isFinal){
  const stripped=value.replace(/["\u0027”’)\]」』】〉》›»]+$/gu,'');
  const ending=stripped.slice(-1);
  if('.!?:;…。！？：；｡,，'.includes(ending))return ending;
  return isFinal?'end':'length';
}

function trimInterChunkTail(buffer,context){
  const frameSize=Math.max(1,Math.round(buffer.sampleRate*.01));
  const frameCount=Math.ceil(buffer.length/frameSize);
  const rms=new Float32Array(frameCount);
  let peak=0;
  for(let frame=0;frame<frameCount;frame+=1){
    const begin=frame*frameSize, end=Math.min(buffer.length,begin+frameSize);
    let sum=0, count=0;
    for(let channel=0;channel<buffer.numberOfChannels;channel+=1){
      const samples=buffer.getChannelData(channel);
      for(let offset=begin;offset<end;offset+=1){sum+=samples[offset]*samples[offset]}
      count+=end-begin;
    }
    rms[frame]=Math.sqrt(sum/Math.max(count,1));
    peak=Math.max(peak,rms[frame]);
  }
  const threshold=Math.max(.0015,peak*.05);
  let lastActive=frameCount-1;
  while(lastActive>=0&&rms[lastActive]<=threshold)lastActive-=1;
  if(lastActive<0)return buffer;
  const retainedTail=Math.round(buffer.sampleRate*.08);
  const cutoff=Math.min(buffer.length,(lastActive+1)*frameSize+retainedTail);
  if(buffer.length-cutoff<Math.round(buffer.sampleRate*.12))return buffer;
  const trimmed=context.createBuffer(buffer.numberOfChannels,cutoff,buffer.sampleRate);
  const fadeSamples=Math.min(Math.round(buffer.sampleRate*.02),cutoff);
  for(let channel=0;channel<buffer.numberOfChannels;channel+=1){
    const output=trimmed.getChannelData(channel);
    output.set(buffer.getChannelData(channel).subarray(0,cutoff));
    for(let index=0;index<fadeSamples;index+=1){
      output[cutoff-fadeSamples+index]*=1-index/Math.max(fadeSamples-1,1);
    }
  }
  return trimmed;
}

async function stopPlayback(showStatus=true){
  generation+=1;
  if(controller){controller.abort();controller=null}
  for(const source of activeSources){try{source.stop()}catch(_){}}
  activeSources=[];
  if(audioContext){await audioContext.close();audioContext=null}
  button.disabled=false;
  if(showStatus)status.textContent='Stopped.';
}

stopButton.onclick=()=>stopPlayback();
form.onsubmit=async e=>{
  e.preventDefault();
  await stopPlayback(false);
  const token=generation;
  const chunks=sentenceChunks(text.value,language.value);
  if(!chunks.length){status.textContent='Enter some text.';return}
  console.info('[Supertonic] chunk plan',{
    sourceChars:text.value.length,
    normalizedChars:text.value.normalize('NFKD').replace(/\s+/gu,' ').trim().length,
    processedLimit:MAX_MODEL_CHARS,
    languageWrapperChars:`<${language.value}></${language.value}>`.length,
    chunks:chunks.map((chunk,index)=>({
      index:index+1,
      chars:chunk.length,
      boundary:splitBoundary(chunk,index===chunks.length-1),
      text:chunk
    }))
  });
  button.disabled=true;
  audioContext=new (window.AudioContext||window.webkitAudioContext)();
  await audioContext.resume();
  const context=audioContext;
  let scheduledUntil=0;
  let audioSeconds=0, generationSeconds=0, trimmedSeconds=0;
  try{
    for(let index=0;index<chunks.length;index+=1){
      if(token!==generation)throw new DOMException('Stopped','AbortError');
      status.textContent=scheduledUntil
        ?`Playing queued audio · synthesizing chunk ${index+1}/${chunks.length}…`
        :`Synthesizing first chunk of ${chunks.length}…`;
      controller=new AbortController();
      const response=await fetch('/v1/speech',{
        method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          input:chunks[index],voice:voice.value,language:language.value,
          speed:Number(speed.value),seed:Number(seed.value)+index,
          chunk_index:index+1,chunk_count:chunks.length,
          source_chars:text.value.length,
          split_boundary:splitBoundary(chunks[index],index===chunks.length-1)
        })
      });
      controller=null;
      if(!response.ok){
        let message=`HTTP ${response.status}`;
        try{message=(await response.json()).error||message}catch(_){}
        throw new Error(message);
      }
      generationSeconds+=Number(response.headers.get('X-Generation-Length-Seconds'))||0;
      const decoded=await context.decodeAudioData(await response.arrayBuffer());
      const buffer=index<chunks.length-1?trimInterChunkTail(decoded,context):decoded;
      audioSeconds+=buffer.duration;
      trimmedSeconds+=decoded.duration-buffer.duration;
      console.info('[Supertonic] chunk ready',{
        index:index+1,
        chunks:chunks.length,
        text:chunks[index],
        rawChars:chunks[index].length,
        processedChars:Number(response.headers.get('X-Text-Length')),
        latentFrames:Number(response.headers.get('X-Latent-Length')),
        audioSeconds:decoded.duration,
        trimmedSeconds:decoded.duration-buffer.duration,
        generationSeconds:Number(response.headers.get('X-Generation-Length-Seconds')),
        rtf:Number(response.headers.get('X-Real-Time-Factor'))
      });
      if(token!==generation)throw new DOMException('Stopped','AbortError');
      const source=context.createBufferSource();
      source.buffer=buffer;
      source.connect(context.destination);
      const startAt=scheduledUntil
        ?Math.max(scheduledUntil,context.currentTime+.02)
        :context.currentTime+.05;
      source.start(startAt);
      scheduledUntil=startAt+buffer.duration;
      activeSources.push(source);
      if(index===chunks.length-1)source.onended=()=>{
        if(token===generation)status.textContent=`Finished ${chunks.length} chunk${chunks.length===1?'':'s'} · audio ${audioSeconds.toFixed(2)} s · trimmed ${trimmedSeconds.toFixed(2)} s between chunks · generation ${generationSeconds.toFixed(2)} s`;
      };
    }
    status.textContent=`Playing ${chunks.length} queued chunk${chunks.length===1?'':'s'} · synthesis complete…`;
  }catch(error){
    if(error.name!=='AbortError')status.textContent=`Error: ${error.message}`;
  }finally{
    controller=null;
    button.disabled=false;
  }
};
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
