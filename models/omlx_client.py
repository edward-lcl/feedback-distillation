"""
Minimal client for the local oMLX server (OpenAI-compatible API).

Auth: reads OMLX_API_KEY (falling back to OPENAI_API_KEY) and sends it as a
Bearer token. If neither is set, no auth header is sent (fine for open servers).
URL: OMLX_URL env var or http://localhost:8000/v1.
"""
import os
import time
import requests

DEFAULT_URL = "http://localhost:8000/v1"


class OmlxClient:
    def __init__(self, api_url: str = None, model: str = None, timeout: int = None,
                 max_retries: int = 10, retry_backoff: float = 5.0):
        self.api_url = (api_url or os.environ.get("OMLX_URL") or DEFAULT_URL).rstrip("/")
        # Model must be a real id on multi-model servers ("default" 404s).
        # Resolution: explicit arg > OMLX_MODEL env > first model on the server.
        self.model = model or os.environ.get("OMLX_MODEL")
        # Timeout: explicit arg > OMLX_TIMEOUT env > 300s. Long max-length
        # generations under load (or a remote teacher over a tunnel) can exceed
        # the old 120s default and trigger a silent retry-storm — see run notes.
        self.timeout = timeout or int(os.environ.get("OMLX_TIMEOUT", "300"))
        # Long labeling runs hit transient server hiccups (streams dropped
        # mid-response → ChunkedEncodingError, model-load races → 5xx). Retry
        # those rather than lose a whole teacher to one flaky call.
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        key = os.environ.get("OMLX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.api_url}/models", headers=self.headers, timeout=10)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def chat(self, prompt: str, max_tokens: int = 150, temperature: float = 0.0) -> str:
        if self.model is None:
            available = self.list_models()
            if not available:
                raise RuntimeError("oMLX server reports no models")
            self.model = available[0]
            print(f"[omlx] no model specified — using first available: {self.model}")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(f"{self.api_url}/chat/completions",
                                     headers=self.headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e  # transient — always retry
            except requests.exceptions.HTTPError as e:
                # Retry server-side 5xx; surface client 4xx immediately.
                if resp.status_code < 500:
                    raise
                last_err = e
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_backoff * (attempt + 1))
        raise last_err
