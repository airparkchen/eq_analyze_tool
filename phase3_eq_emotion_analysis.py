from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", None)


DATA_DIR = Path("/home/parker6/Music/MusicNorm/EQ_exp1_phase3_data")

# Optional focal hypothesis checks.
FOCAL_TRANSITIONS = [("EQ1", "EQ2"), ("EQ1", "EQ15")]

# Target ordering for phase3: good > neutral > bad
EQ_CLASS_MAP = {
    "EQ2": "good",
    "EQ7": "good",
    "EQ1": "neutral",
    "EQ15": "bad",
    "EQ16": "bad",
}
CLASS_SCORE = {"good": 2, "neutral": 1, "bad": 0}

# v0 fixed mapping, independent from v1 tuning.
V0_SCORE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
V0_MODE = "last"  # options: "last", "mean"

RAW_SCORE_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# v1 rebuilt from raw labels (tunable).
V1_CONFIG = {
    "weights": {"positive": 1.0, "neutral": 0.5, "negative": 0.0},
    "window_start": 0,
    # "auto" = use session's actual end (works with variable-length phase3 data)
    "window_end": "auto",
}

# Keep this compact for fast iteration.
V1_TUNING_CANDIDATES = {
    "weights": [
        {"positive": 1.0, "neutral": 0.5, "negative": 0.0},
        {"positive": 1.0, "neutral": 0.25, "negative": 0.0},
        {"positive": 1.0, "neutral": 0.0, "negative": 0.0},
        {"positive": 0.0, "neutral": 0.5, "negative": 1.0},
        {"positive": -1.0, "neutral": 0.5, "negative": 1.0},
    ],
    "windows": [(0, 4), (0, 3), (1, 4), (1, 3), (2, 4), (3, 4)],
}

# Transition table should represent EQ switching events only.
IGNORE_SAME_EQ_TRANSITIONS = True


@dataclass
class Session:
    subject: str
    source_file: str
    subject_session_index: int
    eq: str
    song: str
    start_time: str
    raw: list[str]
    vote: list[str]


def is_single_eq_label(eq_value: object) -> bool:
    return bool(re.match(r"^E\d+$", str(eq_value).strip()))


def normalize_eq(eq: str) -> str:
    match = re.match(r"^E(\d+)$", str(eq).strip())
    if not match:
        return str(eq).strip()
    return f"EQ{int(match.group(1))}"


def infer_subject_name(csv_path: Path) -> str:
    stem = csv_path.stem
    suffix_match = re.match(r"^EQTest2_(.+)$", stem)
    suffix = suffix_match.group(1) if suffix_match else stem

    tokens = [t for t in suffix.split("_") if t]
    alpha_candidates: list[str] = []
    for token in tokens:
        m = re.match(r"^([A-Za-z]+)", token)
        if not m:
            continue
        name = m.group(1).lower()
        alpha_candidates.append(name)

    # Prefer non-system tokens (not eqtest-like), then fall back to first alpha token.
    for name in alpha_candidates:
        if not name.startswith("eqtest") and name not in {"eq", "e"}:
            return name
    if alpha_candidates:
        return alpha_candidates[0]

    alpha_chunks = re.findall(r"[A-Za-z]+", suffix)
    if alpha_chunks:
        return alpha_chunks[-1].lower()

    return suffix.lower()


def resolve_window_end(raw_end_value: object, session_length: int) -> int:
    """
    Resolve window_end against variable-length phase3 sessions.

    Supported values:
    - int > 0: explicit upper bound
    - "auto"/"end"/None: session end
    - int <= 0: also treated as session end for convenience
    """
    if raw_end_value is None:
        return session_length
    if isinstance(raw_end_value, str) and raw_end_value.strip().lower() in {"auto", "end"}:
        return session_length

    end = int(raw_end_value)
    if end <= 0:
        return session_length
    return end


def get_v1_window(config: dict, session_length: int) -> tuple[int, int]:
    start = int(config["window_start"])
    raw_end_value = config.get("window_end", "auto")
    end = resolve_window_end(raw_end_value, session_length)
    if start < 0:
        raise ValueError(f"Invalid v1 window_start: {start}. Must be >= 0")
    if start >= end:
        raise ValueError(f"Invalid v1 window: [{start}:{raw_end_value}) resolved to [{start}:{end})")
    return min(start, session_length), min(end, session_length)


def score_vote_v0(vote_labels: list[str], mode: str = V0_MODE) -> float:
    scored = [V0_SCORE_MAP[label] for label in vote_labels if label in V0_SCORE_MAP]
    if not scored:
        return np.nan
    if mode == "mean":
        return float(np.mean(scored))
    return float(scored[-1])


def score_vote_v1(raw_labels: list[str], config: dict) -> float:
    start, end = get_v1_window(config, len(raw_labels))
    selected = raw_labels[start:end]
    scored = [config["weights"][label] for label in selected if label in config["weights"]]
    if not scored:
        return np.nan
    return float(np.mean(scored))


def score_raw_mean(raw_labels: list[str]) -> float:
    scored = [RAW_SCORE_MAP[label] for label in raw_labels if label in RAW_SCORE_MAP]
    if not scored:
        return np.nan
    return float(np.mean(scored))


def build_session_record(subject: str, source_file: str, subject_session_index: int, rows: list[pd.Series]) -> Session:
    first = rows[0]
    return Session(
        subject=subject,
        source_file=source_file,
        subject_session_index=subject_session_index,
        eq=normalize_eq(first["eq"]),
        song=str(first.get("song", "")),
        start_time=str(first.get("window_start", "")),
        raw=[str(r["raw_prediction"]).strip() for r in rows],
        vote=[str(r["consensus_result"]).strip() for r in rows],
    )


def load_sessions(data_dir: Path) -> list[Session]:
    all_sessions: list[Session] = []
    subject_counts: dict[str, int] = {}
    dropped_multi_eq_rows = 0

    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        valid_mask = df["eq"].map(is_single_eq_label)
        dropped_multi_eq_rows += int((~valid_mask).sum())
        df = df.loc[valid_mask].copy()
        if df.empty:
            continue

        subject = infer_subject_name(csv_path)
        current_rows: list[pd.Series] = []

        for _, row in df.iterrows():
            same_block = bool(
                current_rows
                and row["eq"] == current_rows[-1]["eq"]
                and row.get("song", "") == current_rows[-1].get("song", "")
            )
            if not current_rows or same_block:
                current_rows.append(row)
                continue

            subject_counts[subject] = subject_counts.get(subject, 0) + 1
            all_sessions.append(
                build_session_record(subject, csv_path.name, subject_counts[subject], current_rows)
            )
            current_rows = [row]

        if current_rows:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
            all_sessions.append(
                build_session_record(subject, csv_path.name, subject_counts[subject], current_rows)
            )

    sorted_sessions = sorted(all_sessions, key=lambda s: (s.subject, s.start_time, s.subject_session_index))
    # Attach quick debug info so callers can report filter behavior without changing return type.
    load_sessions.last_dropped_multi_eq_rows = dropped_multi_eq_rows
    return sorted_sessions


def build_session_table(sessions: list[Session], v1_config: dict, v0_mode: str = V0_MODE) -> pd.DataFrame:
    rows = []
    for s in sessions:
        rows.append(
            {
                "subject": s.subject,
                "source_file": s.source_file,
                "subject_session_index": s.subject_session_index,
                "start_time": s.start_time,
                "EQ": s.eq,
                "song": s.song,
                "raw_windows": len(s.raw),
                "vote_windows": len(s.vote),
                "投票v0_mean": score_vote_v0(s.vote, v0_mode),
                "投票v1_mean": score_vote_v1(s.raw, v1_config),
                "Raw_mean": score_raw_mean(s.raw),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["subject", "start_time", "subject_session_index"]
    ).reset_index(drop=True)


def build_eq_summary_table(session_table: pd.DataFrame) -> pd.DataFrame:
    return (
        session_table.groupby(["subject", "EQ"], as_index=False)
        .agg(
            Raw_mean=("Raw_mean", "mean"),
            投票v0_mean=("投票v0_mean", "mean"),
            投票v1_mean=("投票v1_mean", "mean"),
            資料數=("EQ", "size"),
        )
        .sort_values(["subject", "EQ"])
        .reset_index(drop=True)
    )


def get_expected_sign(from_eq: str, to_eq: str) -> float:
    from_class = EQ_CLASS_MAP.get(from_eq)
    to_class = EQ_CLASS_MAP.get(to_eq)
    if from_class is None or to_class is None:
        return np.nan
    delta = CLASS_SCORE[to_class] - CLASS_SCORE[from_class]
    if delta > 0:
        return 1.0
    if delta < 0:
        return -1.0
    return 0.0


def matches_expected(delta_value: float, expected_sign: float) -> float:
    if pd.isna(expected_sign) or expected_sign == 0:
        return np.nan
    return bool(delta_value * expected_sign > 0)


def build_transition_table(session_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in sorted(session_table["subject"].unique()):
        sub = (
            session_table[session_table["subject"] == subject]
            .sort_values(["start_time", "subject_session_index"])
            .reset_index(drop=True)
        )
        for i in range(1, len(sub)):
            prev_row = sub.iloc[i - 1]
            curr_row = sub.iloc[i]
            if IGNORE_SAME_EQ_TRANSITIONS and prev_row["EQ"] == curr_row["EQ"]:
                continue
            expected_sign = get_expected_sign(prev_row["EQ"], curr_row["EQ"])
            rows.append(
                {
                    "subject": subject,
                    "from_session_index": int(prev_row["subject_session_index"]),
                    "to_session_index": int(curr_row["subject_session_index"]),
                    "from_EQ": prev_row["EQ"],
                    "to_EQ": curr_row["EQ"],
                    "transition": f"{prev_row['EQ']}->{curr_row['EQ']}",
                    "from_class": EQ_CLASS_MAP.get(prev_row["EQ"], "unknown"),
                    "to_class": EQ_CLASS_MAP.get(curr_row["EQ"], "unknown"),
                    "expected_sign": expected_sign,
                    "delta_投票v0": curr_row["投票v0_mean"] - prev_row["投票v0_mean"],
                    "delta_投票v1": curr_row["投票v1_mean"] - prev_row["投票v1_mean"],
                    "delta_Raw": curr_row["Raw_mean"] - prev_row["Raw_mean"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "subject",
                "from_session_index",
                "to_session_index",
                "from_EQ",
                "to_EQ",
                "transition",
                "from_class",
                "to_class",
                "expected_sign",
                "delta_投票v0",
                "delta_投票v1",
                "delta_Raw",
                "投票v0_是否變好",
                "投票v1_是否變好",
                "Raw_是否變好",
                "投票v0_是否符合期望",
                "投票v1_是否符合期望",
                "Raw_是否符合期望",
            ]
        )

    df = pd.DataFrame(rows)
    df["投票v0_是否變好"] = df["delta_投票v0"] > 0
    df["投票v1_是否變好"] = df["delta_投票v1"] > 0
    df["Raw_是否變好"] = df["delta_Raw"] > 0
    df["投票v0_是否符合期望"] = df.apply(
        lambda r: matches_expected(r["delta_投票v0"], r["expected_sign"]), axis=1
    )
    df["投票v1_是否符合期望"] = df.apply(
        lambda r: matches_expected(r["delta_投票v1"], r["expected_sign"]), axis=1
    )
    df["Raw_是否符合期望"] = df.apply(
        lambda r: matches_expected(r["delta_Raw"], r["expected_sign"]), axis=1
    )
    return df


def build_transition_stats(transition_table: pd.DataFrame) -> pd.DataFrame:
    if transition_table.empty:
        return pd.DataFrame(
            columns=[
                "subject",
                "transition",
                "次數",
                "投票v0_平均delta",
                "投票v1_平均delta",
                "Raw_平均delta",
                "投票v0_變好率",
                "投票v1_變好率",
                "Raw_變好率",
                "投票v0_符合期望率",
                "投票v1_符合期望率",
                "Raw_符合期望率",
            ]
        )

    per_subject = (
        transition_table.groupby(["subject", "transition"], as_index=False)
        .agg(
            次數=("transition", "size"),
            投票v0_平均delta=("delta_投票v0", "mean"),
            投票v1_平均delta=("delta_投票v1", "mean"),
            Raw_平均delta=("delta_Raw", "mean"),
            投票v0_變好率=("投票v0_是否變好", "mean"),
            投票v1_變好率=("投票v1_是否變好", "mean"),
            Raw_變好率=("Raw_是否變好", "mean"),
            投票v0_符合期望率=("投票v0_是否符合期望", "mean"),
            投票v1_符合期望率=("投票v1_是否符合期望", "mean"),
            Raw_符合期望率=("Raw_是否符合期望", "mean"),
        )
    )
    overall = (
        transition_table.groupby(["transition"], as_index=False)
        .agg(
            次數=("transition", "size"),
            投票v0_平均delta=("delta_投票v0", "mean"),
            投票v1_平均delta=("delta_投票v1", "mean"),
            Raw_平均delta=("delta_Raw", "mean"),
            投票v0_變好率=("投票v0_是否變好", "mean"),
            投票v1_變好率=("投票v1_是否變好", "mean"),
            Raw_變好率=("Raw_是否變好", "mean"),
            投票v0_符合期望率=("投票v0_是否符合期望", "mean"),
            投票v1_符合期望率=("投票v1_是否符合期望", "mean"),
            Raw_符合期望率=("Raw_是否符合期望", "mean"),
        )
    )
    overall["subject"] = "ALL"
    cols = [
        "subject",
        "transition",
        "次數",
        "投票v0_平均delta",
        "投票v1_平均delta",
        "Raw_平均delta",
        "投票v0_變好率",
        "投票v1_變好率",
        "Raw_變好率",
        "投票v0_符合期望率",
        "投票v1_符合期望率",
        "Raw_符合期望率",
    ]
    return pd.concat([per_subject[cols], overall[cols]], ignore_index=True).sort_values(
        ["subject", "transition"]
    ).reset_index(drop=True)


def build_focal_transition_table(transition_stats: pd.DataFrame) -> pd.DataFrame:
    labels = [f"{a}->{b}" for a, b in FOCAL_TRANSITIONS]
    return transition_stats[transition_stats["transition"].isin(labels)].copy().reset_index(drop=True)


def build_v0_raw_chunk_table(transition_stats: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "subject",
        "transition",
        "次數",
        "投票v0_平均delta",
        "Raw_平均delta",
        "投票v0_符合期望率",
        "Raw_符合期望率",
        "投票v0_變好率",
        "Raw_變好率",
    ]
    if transition_stats.empty:
        return pd.DataFrame(columns=cols)
    return transition_stats[cols].copy().sort_values(["subject", "transition"]).reset_index(drop=True)


def format_weight_label(weights: dict) -> str:
    p = weights.get("positive", np.nan)
    n = weights.get("neutral", np.nan)
    g = weights.get("negative", np.nan)
    return f"P{p}_N{n}_G{g}"


def build_tuning_table(sessions: list[Session], v0_mode: str = V0_MODE) -> pd.DataFrame:
    rows = []
    for weights in V1_TUNING_CANDIDATES["weights"]:
        for start, end in V1_TUNING_CANDIDATES["windows"]:
            cfg = {"weights": weights, "window_start": start, "window_end": end}
            session_table = build_session_table(sessions, cfg, v0_mode)
            transition_table = build_transition_table(session_table)
            has_transition_cols = {"from_EQ", "to_EQ"}.issubset(set(transition_table.columns))

            if transition_table.empty or not has_transition_cols:
                rows.append(
                    {
                        "投票v1_權重(作用於Raw)": format_weight_label(weights),
                        "Raw_window": f"[{start}:{end})",
                        "transition_次數": 0,
                        "投票v1_符合期望率": np.nan,
                        "E1->E2_次數": 0,
                        "投票v1_E1->E2平均Delta": np.nan,
                        "投票v1_E1->E2變好率": np.nan,
                    }
                )
                continue

            e1_to_e2 = transition_table[
                (transition_table["from_EQ"] == "EQ1") & (transition_table["to_EQ"] == "EQ2")
            ].copy()

            rows.append(
                {
                    "投票v1_權重(作用於Raw)": format_weight_label(weights),
                    "Raw_window": f"[{start}:{end})",
                    "transition_次數": int(len(transition_table)),
                    "投票v1_符合期望率": float(transition_table["投票v1_是否符合期望"].mean()),
                    "E1->E2_次數": int(len(e1_to_e2)),
                    "投票v1_E1->E2平均Delta": (
                        float(e1_to_e2["delta_投票v1"].mean()) if not e1_to_e2.empty else np.nan
                    ),
                    "投票v1_E1->E2變好率": (
                        float(e1_to_e2["投票v1_是否變好"].mean()) if not e1_to_e2.empty else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["投票v1_符合期望率", "投票v1_E1->E2變好率", "投票v1_E1->E2平均Delta"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def main() -> None:
    sessions = load_sessions(DATA_DIR)
    if not sessions:
        raise SystemExit(f"No csv files found in {DATA_DIR}")

    session_table = build_session_table(sessions, V1_CONFIG, V0_MODE)
    eq_summary = build_eq_summary_table(session_table)
    transition_table = build_transition_table(session_table)
    transition_stats = build_transition_stats(transition_table)
    focal_transition_table = build_focal_transition_table(transition_stats)
    v0_raw_chunk_table = build_v0_raw_chunk_table(transition_stats)
    tuning_table = build_tuning_table(sessions, V0_MODE)

    print(f"Phase3 data dir: {DATA_DIR}")
    print(f"v0 mode: {V0_MODE}")
    print(f"v1 config: {V1_CONFIG}")
    print(f"dropped multi-EQ rows: {getattr(load_sessions, 'last_dropped_multi_eq_rows', 0)}")
    print()
    print("Session Table")
    print(session_table.to_string(index=False))
    print()
    print("EQ Summary Table")
    print(eq_summary.to_string(index=False))
    print()
    print("Transition Table (adjacent sessions)")
    print(transition_table.to_string(index=False))
    print()
    print("Transition Stats (by subject + ALL)")
    print(transition_stats.to_string(index=False))
    print()
    print("Focal Transition Table")
    print(focal_transition_table.to_string(index=False))
    print()
    print("v0 + Raw Chunk Table")
    print(v0_raw_chunk_table.to_string(index=False))
    print()
    print("v1 Tuning Table")
    print(tuning_table.to_string(index=False))


if __name__ == "__main__":
    main()
