#!/usr/bin/env python3
"""Push proxy for the PD protocol (kv_compare/pd/connector.py): one prefill engine, N decode engines.

Each request goes to the prefill engine (max_tokens 1, kv_transfer_params telling it where to send the KV and with
which codec) and, at the same time, to one decode engine (told to wait for that transfer); the client gets the
decode engine's response. The decode engine is chosen round-robin, by fewest requests in flight, or - when the
request carries a "conversation_id" (vLLM's multi-turn benchmark, --send-conversation-id) and the policy is
"conversation" - the engine that served the conversation before.

  python scripts/pd_proxy.py --port 8000 --prefill http://127.0.0.1:8100 \
      --decode http://127.0.0.1:8201=127.0.0.1:9201 --decode http://127.0.0.1:8202=127.0.0.1:9202 --log proxy.jsonl

Control (the benchmark driver):  POST /t/config {"codec": "ours-hbh", "verify": false, "mode": "pd" | "local",
  "policy": "round_robin" | "least_loaded" | "conversation", "reset_conversations": true}
  mode local: no transfer; requests go straight to the decode engines, which prefill themselves (the colocated
  baseline on the same GPUs).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
import uuid

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TIMEOUT = aiohttp.ClientTimeout(total=6 * 3600)


class Proxy:
    def __init__(self, prefill, decodes, policy, codec, log_path):
        self.prefill = prefill.rstrip("/")
        self.decodes = [(u.rstrip("/"), kv) for u, kv in decodes]
        self.policy = policy
        self.cfg = {"codec": codec, "verify": False, "mode": "pd"}
        self.rr = itertools.cycle(range(len(self.decodes)))
        self.inflight = [0] * len(self.decodes)
        self.conv = {}
        self.log_path = log_path
        self.session = None

    def log(self, rec):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def pick(self, conv):
        if self.policy == "conversation" and conv is not None and conv in self.conv:
            return self.conv[conv]
        if self.policy == "least_loaded":
            i = min(range(len(self.decodes)), key=lambda j: (self.inflight[j], j))
        else:
            i = next(self.rr)
        if conv is not None:
            self.conv[conv] = i
        return i

    async def handle(self, raw: Request, path: str):
        body = await raw.json()
        conv = body.pop("conversation_id", None)
        key = uuid.uuid4().hex
        i = self.pick(conv)
        durl, kvaddr = self.decodes[i]
        cfg = dict(self.cfg)
        rec = {"key": key, "dest": i, "codec": cfg["codec"], "mode": cfg["mode"], "conv": conv, "t_in": time.time()}
        headers = {"X-Request-Id": key}
        self.inflight[i] += 1
        d_body = dict(body)
        p_task = None
        if cfg["mode"] == "pd":
            p_body = dict(body, max_tokens=1, stream=False)
            p_body.pop("stream_options", None)
            if "max_completion_tokens" in p_body:
                p_body["max_completion_tokens"] = 1
            p_body["kv_transfer_params"] = {"t_send": True, "t_key": key, "t_dest": kvaddr, "t_codec": cfg["codec"],
                                            "t_verify": bool(cfg["verify"]), "t_conv": conv}
            d_body["kv_transfer_params"] = {"t_recv": True, "t_key": key}

            async def prefill():
                async with self.session.post(self.prefill + path, json=p_body, headers=headers) as r:
                    txt = await r.read()
                    rec["t_prefill_done"] = time.time()
                    if r.status != 200:
                        raise RuntimeError(f"prefill engine {r.status}: {txt[:300]!r}")
            p_task = asyncio.create_task(prefill())

        resp = await self.session.post(durl + path, json=d_body, headers=headers)
        if resp.status != 200:
            txt = await resp.read()
            self.inflight[i] -= 1
            rec.update(error=f"decode engine {resp.status}: {txt[:300]!r}", t_out=time.time())
            self.log(rec)
            if p_task is not None:
                p_task.cancel()
            return JSONResponse(status_code=resp.status, content={"error": txt.decode(errors="replace")})

        async def stream():
            first = True
            try:
                if p_task is not None:
                    # a prefill failure would leave the decode engine waiting for a transfer that never comes:
                    # watch it while streaming and close the decode request (vLLM aborts it) if it fails
                    async def watch():
                        try:
                            await p_task
                        except Exception as exc:
                            rec["error"] = repr(exc)[:300]
                            resp.close()
                    w = asyncio.create_task(watch())
                async for chunk in resp.content.iter_any():
                    if first:
                        rec["t_first_byte"] = time.time()
                        first = False
                    yield chunk
            finally:
                resp.release()
                self.inflight[i] -= 1
                if p_task is not None:
                    try:
                        await p_task
                    except Exception as exc:
                        rec.setdefault("error", repr(exc)[:300])
                    w.cancel()
                rec["t_out"] = time.time()
                self.log(rec)
        media = resp.headers.get("content-type", "application/json")
        return StreamingResponse(stream(), media_type=media)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", action="append", required=True, help="http_url=kv_host:kv_port (repeat)")
    ap.add_argument("--policy", default="round_robin", choices=["round_robin", "least_loaded", "conversation"])
    ap.add_argument("--codec", default="raw")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    decodes = [tuple(d.split("=", 1)) for d in a.decode]
    px = Proxy(a.prefill, decodes, a.policy, a.codec, a.log)
    app = FastAPI()

    @app.on_event("startup")
    async def _start():
        px.session = aiohttp.ClientSession(timeout=TIMEOUT, connector=aiohttp.TCPConnector(limit=0))

    @app.on_event("shutdown")
    async def _stop():
        await px.session.close()

    @app.post("/v1/completions")
    async def completions(raw: Request):
        return await px.handle(raw, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat(raw: Request):
        return await px.handle(raw, "/v1/chat/completions")

    @app.post("/t/config")
    async def config(raw: Request):
        body = await raw.json()
        px.cfg.update({k: v for k, v in body.items() if k in ("codec", "verify", "mode")})
        if body.get("policy") in ("round_robin", "least_loaded", "conversation"):
            px.policy = body["policy"]
        if body.get("reset_conversations"):
            px.conv.clear()
        return dict(px.cfg, policy=px.policy)

    @app.get("/t/status")
    async def status():
        return {"cfg": px.cfg, "inflight": px.inflight, "decodes": px.decodes, "prefill": px.prefill,
                "policy": px.policy, "conversations": len(px.conv)}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/models")
    async def models():
        async with px.session.get(px.decodes[0][0] + "/v1/models") as r:
            return JSONResponse(status_code=r.status, content=await r.json())

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
