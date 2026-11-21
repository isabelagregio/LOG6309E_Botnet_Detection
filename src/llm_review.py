from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

# Same features as the ML baseline.
NUMERIC = ["dur", "stos", "dtos", "tot_pkts", "tot_bytes", "src_bytes"]
CATEGORICAL = ["proto", "dir", "state"]
FEATURES = NUMERIC + CATEGORICAL


def prepare_requests(csv_paths: list[Path], prompt_path: Path) -> list[dict]:
    """Use only the feature allowlist in prompts; retain selection memberships.

    One flow may appear in multiple budgets/strategies. Its request is prepared
    once, with every source file retained for later evaluation.
    """
    prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("The prompt file is empty")
    prompt_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    requests = {}

    for csv_path in sorted(set(map(Path, csv_paths))):
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"flow_id", "scenario", *FEATURES}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
            seen = set()
            for line, row in enumerate(reader, start=2):
                flow_id = row["flow_id"]
                if not flow_id or flow_id in seen:
                    raise ValueError(f"{csv_path}:{line}: empty or duplicate flow_id")
                seen.add(flow_id)
                features = {}
                for name in FEATURES:
                    value = row[name]
                    if value is None or not value.strip():
                        features[name] = None
                    elif name in NUMERIC:
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError(f"{csv_path}:{line}: nonfinite {name}")
                        features[name] = number
                    else:
                        features[name] = value
                scenario = int(row["scenario"])
                messages = [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(features, allow_nan=False),
                    },
                ]
                if flow_id in requests:
                    previous = requests[flow_id]
                    if previous["scenario"] != scenario or previous["messages"] != messages:
                        raise ValueError(f"Conflicting features/scenario for {flow_id}")
                    previous["source_files"].append(str(csv_path))
                else:
                    requests[flow_id] = {
                        "flow_id": flow_id,
                        "scenario": scenario,
                        "source_files": [str(csv_path)],
                        "prompt_id": prompt_id,
                        "messages": messages,
                    }
    return list(requests.values())


def make_cache_key(
    request: dict, *, backend: str, model: str, generation_settings: dict
) -> str:
    """Identify the exact LLM input and configuration, independent of selection."""
    if not backend.strip() or not model.strip():
        raise ValueError("Cache entries need a backend and model identifier")
    payload = {
        "backend": backend,
        "model": model,
        "generation_settings": generation_settings,
        "messages": request["messages"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_cached_responses(path: Path) -> dict[str, dict]:
    """Index the first completed response for each configuration/input pair.

    Invalid model replies are retained to avoid silently retrying until a valid
    answer appears. Failed calls remain in the log but are not reusable replies.
    """
    path = Path(path)
    if not path.exists():
        return {}
    cache = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid cache JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a response object")
            key = record.get("cache_key")
            if (key and isinstance(record.get("response"), str)
                    and record.get("error") in (None, "invalid_response")):
                cache.setdefault(key, record)
    return cache


def save_response(
    output_path: Path,
    request: dict,
    *,
    model: str,
    response: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    response_time_seconds: float | None,
    error: str | None = None,
    backend: str | None = None,
    generation_settings: dict | None = None,
    response_metadata: dict | None = None,
    cache_hit: bool = False,
) -> dict:
    """Append an actual response and its measurements as a single JSONL record.

    Unknown token counts remain null. Invalid responses remain unclassified;
    this helper does not choose an error fallback or a prediction fusion rule.
    """
    for count in (input_tokens, output_tokens):
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Token counts must be nonnegative integers or None")
    if response_time_seconds is not None:
        if not math.isfinite(response_time_seconds) or response_time_seconds < 0:
            raise ValueError("Response time must be finite and nonnegative or None")
    prediction = None
    if error is None:
        label = response.strip().lower() if response is not None else ""
        prediction = {"normal": 0, "botnet": 1}.get(label)
        if prediction is None:
            error = "invalid_response"
    record = {
        "flow_id": request["flow_id"],
        "scenario": request["scenario"],
        "model": model,
        "prompt_id": request["prompt_id"],
        "response": response,
        "prediction": prediction,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response_time_seconds": response_time_seconds,
        "error": error,
        "cache_hit": cache_hit,
    }
    if response_metadata is not None:
        record["response_metadata"] = response_metadata
    if backend is not None:
        settings = generation_settings if generation_settings is not None else {}
        record.update({
            "backend": backend,
            "generation_settings": settings,
            "cache_key": make_cache_key(
                request, backend=backend, model=model, generation_settings=settings
            ),
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare network flows for LLM review.")
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    requests = prepare_requests(args.csv_files, args.prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve existing preparation runs instead of silently overwriting them.
    with args.output.open("x", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, allow_nan=False) + "\n")
    memberships = sum(len(r["source_files"]) for r in requests)
    print(f"Prepared {len(requests)} unique flows from {memberships} selection rows.")
    print(f"Requests saved to {args.output}. No LLM calls were made.")


if __name__ == "__main__":
    main()
