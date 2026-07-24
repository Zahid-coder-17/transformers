import sys
import os
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from zahidgpt import generate

if HAS_FASTAPI:
    app = FastAPI(
        title="ZahidGPT REST API",
        description="REST API interface for Modular Transformer & Multi-Corpus LLM (English, Arabic, Code)",
        version="0.1.0"
    )

    class GenerateRequest(BaseModel):
        prompt: str = "Once upon a time"
        model_type: str = "multicorpus"
        max_new_tokens: int = 150
        temperature: float = 0.8
        top_k: int = 40
        top_p: float = 0.9

    @app.get("/")
    def root():
        return {"status": "ok", "message": "ZahidGPT LLM API is online"}

    @app.get("/health")
    def health():
        return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

    @app.post("/generate")
    def api_generate(req: GenerateRequest):
        try:
            start_time = time.time()
            output = generate(
                prompt=req.prompt,
                model_type=req.model_type,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p
            )
            elapsed = time.time() - start_time
            return {
                "prompt": req.prompt,
                "model_type": req.model_type,
                "generated_text": output,
                "latency_seconds": round(elapsed, 4)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def run_server(host="0.0.0.0", port=8000):
        uvicorn.run(app, host=host, port=port)

else:
    class SimpleAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ["/", "/health"]:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                res = {"status": "healthy", "gpu_available": torch.cuda.is_available()}
                self.wfile.write(json.dumps(res).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/generate":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    prompt = data.get("prompt", "Once upon a time")
                    model_type = data.get("model_type", "multicorpus")
                    max_new_tokens = int(data.get("max_new_tokens", 150))
                    temp = float(data.get("temperature", 0.8))
                    top_k = int(data.get("top_k", 40))
                    top_p = float(data.get("top_p", 0.9))

                    start = time.time()
                    out = generate(prompt, model_type, max_new_tokens, temp, top_k, top_p)
                    elapsed = time.time() - start

                    res = {
                        "prompt": prompt,
                        "model_type": model_type,
                        "generated_text": out,
                        "latency_seconds": round(elapsed, 4)
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def run_server(host="0.0.0.0", port=8000):
        print(f"Starting lightweight HTTP REST API server on http://{host}:{port} ...")
        httpd = HTTPServer((host, port), SimpleAPIHandler)
        httpd.serve_forever()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port=port)
