#!/usr/bin/env python3
"""Serial/continuous mixed-text-and-image API parity test (stdlib only)."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
import zlib

def png(rgb):
    def chunk(t, b):
        return struct.pack(">I",len(b))+t+b+struct.pack(">I",zlib.crc32(t+b))
    rows=(b"\x00"+bytes(rgb)*112)*112
    return b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",112,112,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(rows))+chunk(b"IEND",b"")

def cases(model, max_tokens=192):
    """28 stable text/image prompts; c16 can genuinely reach 16 active lanes."""
    def image(rgb):
        return "data:image/png;base64," + base64.b64encode(png(rgb)).decode()
    red, blue = image((238, 38, 38)), image((20, 40, 238))
    topics = [
        "concurrency in Rust", "HTTP request routing", "SQLite transactions",
        "JSON parsing", "async programming", "data structures", "API versioning",
        "database indexes", "memory safety", "network latency", "CI testing",
        "GPU inference batching", "error handling", "tokenization",
        "cache invalidation", "structured tool calling", "context windows",
        "garbage collection", "streaming responses", "scheduler fairness",
    ]
    text_prompts = [
        (f"t{i+1}", "Write sixteen separately numbered short sentences about "
         f"{topic}. Every sentence must mention a different specific detail.")
        for i, topic in enumerate(topics)
    ]
    image_prompts = [
        (f"v{i+1}", [
            {"type": "text", "text":
                "Identify the dominant color, then write twelve numbered "
                "short observations describing that color. Do not guess objects."},
            {"type": "image_url", "image_url": {"url": red if i % 2 == 0 else blue}},
        ])
        for i in range(8)
    ]
    # Keep vision arrivals interleaved with text. Image-only batches do not
    # prove that image-conditioned KV can share a text decode Forward.
    prompts = []
    for i in range(20):
        prompts.append(text_prompts[i])
        if i < 8:
            prompts.append(image_prompts[i])
    return [
        {"id": key, "request": {
            "model": model, "temperature": 0,
            "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }}
        for key, prompt in prompts
    ]

def request(url,body=None):
    b=json.dumps(body).encode() if body is not None else None
    req=Request(url,data=b,headers={"Content-Type":"application/json"},method="POST" if b else "GET")
    with urlopen(req,timeout=360) as resp:
        result=json.load(resp)
    if not isinstance(result,dict) or "error" in result:
        raise RuntimeError(f"Bad API response: {result}")
    return result

def normalized(response):
    choices=response.get("choices")
    if not isinstance(choices,list) or not choices:
        raise RuntimeError("Missing choices in response")
    c=choices[0]; m=c.get("message")
    if not isinstance(m,dict):
        raise RuntimeError("Missing message")
    calls=[]
    for t in m.get("tool_calls") or []:
        f=t.get("function") or {}; args=f.get("arguments") or ""
        try: args=json.loads(args)
        except (ValueError,TypeError): pass
        calls.append({"name":f.get("name"),"arguments":args})
    return {"content":m.get("content") or "","tool_calls":calls,"finish_reason":c.get("finish_reason")}

def digest(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()

def run_one(url,c):
    start=time.perf_counter()
    r=request(url+"/v1/chat/completions",c["request"])
    return {"id":c["id"],"output":normalized(r),"ms":round((time.perf_counter()-start)*1000,1),"usage":r.get("usage")}

def compare(gold,current):
    errors=[]
    for field in ("schema","model","hashes"):
        if gold.get(field)!=current.get(field): errors.append(field+" mismatch")
    a={x["id"]:x["output"] for x in gold["results"]}
    b={x["id"]:x["output"] for x in current["results"]}
    for key in sorted(set(a)|set(b)):
        if a.get(key)!=b.get(key): errors.append("Output mismatch: "+key)
    return errors

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-url",default="http://127.0.0.1:18080")
    p.add_argument("--model",default="big")
    p.add_argument("--concurrency",type=int,default=1)
    p.add_argument("--max-tokens",type=int,default=192)
    p.add_argument("--out",type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record",action="store_true")
    mode.add_argument("--compare",type=Path)
    a=p.parse_args()
    if not 1<=a.concurrency<=16 or (a.record and a.concurrency!=1):
        p.error("Record requires c1; valid concurrency 1..16")
    if not 1 <= a.max_tokens <= 2048:
        p.error("--max-tokens must be 1..2048")
    url=a.base_url.rstrip("/")
    models=request(url+"/v1/models")
    if a.model not in [m.get("id") for m in models.get("data",[])]:
        raise RuntimeError("Model not exposed by /v1/models")
    scenarios=cases(a.model, a.max_tokens)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results=list(pool.map(lambda c:run_one(url,c),scenarios))
    elapsed=time.perf_counter()-start
    total_tokens=sum((row.get("usage") or {}).get("completion_tokens") or 0 for row in results)
    report={"schema":"dfm-v41-parity/1","model":a.model,"concurrency":a.concurrency,"elapsed_s":round(elapsed,3),
            "completion_tokens_reported":total_tokens,
            "aggregate_toks_per_s":round(total_tokens/elapsed,2) if elapsed else None,
            "hashes":{c["id"]:digest(c["request"]) for c in scenarios},"results":results}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    errors=compare(json.loads(a.compare.read_text()),report) if a.compare else []
    print("Results:",a.out,"scenarios:",len(scenarios),"elapsed:",round(elapsed,2),"tokens:",total_tokens,"mismatches:",errors)
    return 1 if errors else 0

if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as e:
        print("API gate error:",e)
        raise SystemExit(2) from e
