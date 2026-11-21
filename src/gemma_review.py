from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values

from src.llm_review import FEATURES, load_cached_responses, make_cache_key, save_response

MODEL = "gemma-4-31b-it"
BACKEND = "google-gemini-v1beta"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def get_api_key(env_file: Path | None = None) -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = ENV_FILE if env_file is None else Path(env_file)
    if not path.is_file():
        return ""
    values = dotenv_values(path, interpolate=False)
    return (values.get("GEMINI_API_KEY") or "").strip()


def make_payload(request: dict, generation_settings: dict) -> dict:
    messages = request["messages"]
    if len(messages) != 2 or [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("Expected one instruction and one flow message")
    features = json.loads(messages[1]["content"])
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        raise ValueError("The flow message must contain exactly the nine baseline features")
    return {
        "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
        "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
        "generationConfig": generation_settings,
    }


def parse_response(data: dict) -> dict:
    candidates = data.get("candidates", [])
    candidate = candidates[0] if candidates else {}
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    usage = data.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount")
    total_tokens = usage.get("totalTokenCount")
    output_tokens = None
    if type(input_tokens) is int and type(total_tokens) is int:
        if 0 <= input_tokens <= total_tokens:
            # Include thinking tokens, not just the final label.
            output_tokens = total_tokens - input_tokens
    finish_reason = candidate.get("finishReason")
    return {
        "response": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": "invalid_response" if finish_reason not in (None, "STOP") else None,
        "response_metadata": {
            "model_version": data.get("modelVersion"),
            "response_id": data.get("responseId"),
            "finish_reason": finish_reason,
            "prompt_feedback": data.get("promptFeedback"),
            "usage": usage,
        },
    }


def call_gemma(
    request: dict, api_key: str, generation_settings: dict, timeout: float = 120
) -> dict:
    payload = make_payload(request, generation_settings)
    http_request = Request(
        API_URL, data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urlopen(http_request, timeout=timeout) as response:
        return parse_response(json.loads(response.read()))


def run_reviews(
    requests: list[dict], output_path: Path, *, api_key: str,
    generation_settings: dict, max_new_requests: int,
    requests_per_minute: float = 30, timeout: float = 120,
) -> dict:
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY before executing requests")
    if max_new_requests <= 0 or not 0 < requests_per_minute <= 30:
        raise ValueError("Use a positive call limit and a request rate between 0 and 30")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    output_path = Path(output_path)
    cache = load_cached_responses(output_path)
    recorded = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if (isinstance(row.get("response"), str)
                        and row.get("error") in (None, "invalid_response")):
                    recorded.add((row["flow_id"], row.get("cache_key")))
    config = {"backend": BACKEND, "model": MODEL, "generation_settings": generation_settings}
    stats = {"new_calls": 0, "cache_hits": 0, "already_recorded": 0, "failed_calls": 0}
    next_call = 0.0
    for request in requests:
        make_payload(request, generation_settings)
        key = make_cache_key(request, **config)
        identity = (request["flow_id"], key)
        if identity in recorded:
            stats["already_recorded"] += 1
            continue
        if key in cache:
            previous = cache[key]
            save_response(
                output_path, request, **config, response=previous["response"],
                input_tokens=previous["input_tokens"], output_tokens=previous["output_tokens"],
                response_time_seconds=previous["response_time_seconds"],
                error=previous["error"], response_metadata=previous.get("response_metadata"),
                cache_hit=True,
            )
            recorded.add(identity)
            stats["cache_hits"] += 1
            continue
        if stats["new_calls"] >= max_new_requests:
            break
        time.sleep(max(0.0, next_call - time.monotonic()))
        started = time.perf_counter()
        error = None
        try:
            result = call_gemma(request, api_key, generation_settings, timeout)
        except HTTPError as exc:
            error = f"http_{exc.code}"
        except (URLError, TimeoutError):
            error = "network_error"
        except json.JSONDecodeError:
            error = "invalid_api_json"
        elapsed = time.perf_counter() - started
        next_call = time.monotonic() + 60 / requests_per_minute
        stats["new_calls"] += 1
        if error:
            result = {
                "response": None, "input_tokens": None, "output_tokens": None,
                "error": error,
            }
        record = save_response(
            output_path, request, **config, **result, response_time_seconds=elapsed,
        )
        if error:
            stats["failed_calls"] += 1
            break
        cache[key] = record
        recorded.add(identity)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Review prepared flows with hosted Gemma 4 31B.")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--max-output-tokens", required=True, type=int)
    parser.add_argument("--thinking-level", required=True, choices=["minimal", "high"])
    parser.add_argument("--max-new-requests", type=int)
    parser.add_argument("--requests-per-minute", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("temperature must be between 0 and 2")
    if args.max_output_tokens <= 0:
        parser.error("max-output-tokens must be positive")
    settings = {
        "temperature": args.temperature, "maxOutputTokens": args.max_output_tokens,
        "thinkingConfig": {"thinkingLevel": args.thinking_level},
    }
    requests = [
        json.loads(line) for line in args.requests.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for request in requests:
        make_payload(request, settings)
    print(f"Model: {MODEL}; prepared flows: {len(requests)}; settings: {settings}")
    if not args.execute:
        print("Preview only. Add --execute and --max-new-requests to make API calls.")
        return
    if args.max_new_requests is None or args.max_new_requests <= 0:
        parser.error("execution requires a positive --max-new-requests")
    api_key = get_api_key()
    if not api_key:
        parser.error("set GEMINI_API_KEY in your environment or the repository .env file")
    stats = run_reviews(
        requests, args.output, api_key=api_key, generation_settings=settings,
        max_new_requests=args.max_new_requests, requests_per_minute=args.requests_per_minute,
        timeout=args.timeout,
    )
    print(json.dumps(stats, indent=2))
    if stats["failed_calls"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
