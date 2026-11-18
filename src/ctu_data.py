from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NUMERIC = ["dur", "stos", "dtos", "tot_pkts", "tot_bytes", "src_bytes"]
CATEGORICAL = ["proto", "dir", "state"]
FEATURES = NUMERIC + CATEGORICAL
OUTPUT_COLUMNS = ["flow_id", "scenario", "true_label", *FEATURES]


def find_scenarios(data_dir: Path | str = Path("data/raw")) -> dict[int, Path]:
    """Map CTU-13 scenario numbers to their respective parquet files."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    found: dict[int, Path] = {}
    for path in sorted(root.rglob("*.parquet")):
        match = re.match(r"^(\d{1,2})[-_]", path.name)
        if match and 1 <= int(match.group(1)) <= 13:
            scenario = int(match.group(1))
            if scenario in found:
                raise ValueError(f"Duplicate parquet for scenario {scenario}: {found[scenario]} and {path}")
            found[scenario] = path
    if not found:
        raise FileNotFoundError(f"No CTU-13 scenario parquet files found in {root}")
    return dict(sorted(found.items()))


def choose_column(columns, name: str) -> str:
    matches = [c for c in columns if c.lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected a single column {name!r}; found {list(columns)}")
    return matches[0]


def decode_labels(labels: pd.Series) -> pd.Series:
    """Map only explicit From-Normal/From-Botnet flow labels to 0/1.

    Other labels (including Background) are excluded, not assigned to Normal.
    """
    clean = labels.astype("string").fillna("").str.strip().str.lower()
    botnet = clean.str.contains(r"(?:^|=)from-botnet(?:-|$)", regex=True, na=False)
    normal = clean.str.contains(r"(?:^|=)from-normal(?:-|$)", regex=True, na=False)
    result = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    result.loc[botnet & ~normal] = 1
    result.loc[normal & ~botnet] = 0
    return result


def normalize_chunk(chunk: pd.DataFrame, scenario: int, first_row: int) -> pd.DataFrame:
    label_column = choose_column(chunk.columns, "label")
    labels = decode_labels(chunk[label_column])
    indices = np.flatnonzero(labels.notna().to_numpy())
    if len(indices) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    subset = chunk.iloc[indices]
    output = pd.DataFrame(index=np.arange(len(subset)))
    output["flow_id"] = [f"s{scenario:02d}_r{first_row + int(i)}" for i in indices]
    output["scenario"] = scenario
    output["true_label"] = labels.iloc[indices].astype("int8").to_numpy()
    for name in NUMERIC:
        column = choose_column(chunk.columns, name)
        output[name] = pd.to_numeric(subset[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy()
    for name in CATEGORICAL:
        column = choose_column(chunk.columns, name)
        output[name] = (
            subset[column].astype("string").fillna("missing").astype(str)
            .str.strip().str.lower().replace("", "missing").astype(str).to_numpy()
        )
    return output[OUTPUT_COLUMNS]


def read_scenario(path: Path | str, scenario: int, batch_size: int = 100_000,
                  max_per_class: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Load labeled flows in batches and return rows plus audited class counts.

    The final experiment uses all labeled rows (max_per_class=None).
    Parameters remain compatible with previously generated pipeline modules.
    """
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    wanted = [choose_column(columns, name) for name in ["label", *FEATURES]]
    count = {0: 0, 1: 0}
    raw_rows = 0
    pieces = []
    rng = np.random.default_rng(seed + scenario)
    priorities = {}

    for batch in parquet.iter_batches(columns=wanted, batch_size=batch_size):
        chunk = batch.to_pandas()
        part = normalize_chunk(chunk, scenario, raw_rows)
        raw_rows += len(chunk)
        if part.empty:
            continue
        for cls in (0, 1):
            count[cls] += int((part.true_label == cls).sum())
        if max_per_class is None:
            pieces.append(part)
        else:
            if max_per_class <= 0:
                raise ValueError("max_per_class must be > 0")
            part = part.copy()
            part["_random_priority"] = rng.random(len(part))
            for cls in (0, 1):
                subset = part.loc[part.true_label == cls]
                if subset.empty:
                    continue
                candidate = pd.concat([priorities.get(cls, subset.iloc[:0]), subset], ignore_index=True)
                priorities[cls] = candidate.nsmallest(max_per_class, "_random_priority")

    if max_per_class is not None:
        pieces = [part.drop(columns="_random_priority") for part in priorities.values()]
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not frame.empty:
        frame["true_label"] = frame.true_label.astype("int8")
    info = {
        "scenario": scenario, "raw_rows": raw_rows,
        "labeled_normal": count[0], "labeled_botnet": count[1],
        "excluded_unlabeled": raw_rows - count[0] - count[1],
        "retained_rows": len(frame),
        "retained_normal": int((frame.true_label == 0).sum()),
        "retained_botnet": int((frame.true_label == 1).sum()),
    }
    return frame, info
