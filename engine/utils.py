from __future__ import annotations

import re
import numpy as np
import pandas as pd


def is_probably_id_name(col_name: str) -> bool:
    """
    列名からIDっぽさをざっくり判定する
    """
    patterns = [
        r"\bid\b",
        r"ID",
        r"id",
        r"番号",
        r"学籍",
        r"受験",
        r"コード",
        r"code",
        r"キー",
        r"key",
    ]
    return any(re.search(p, str(col_name)) for p in patterns)


def safe_unique_ratio(series: pd.Series) -> float:
    """
    ユニーク率を返す
    """
    if len(series) == 0:
        return 0.0
    return series.nunique(dropna=True) / len(series)


def normalize_string_series(series: pd.Series) -> pd.Series:
    """
    ID変換前の正規化用
    """
    return series.astype("string").str.strip()


def make_prefixed_ids(n: int, prefix: str = "SID") -> list[str]:
    """
    SID_000001 のような人工IDを生成する
    """
    width = max(6, len(str(n)))
    return [f"{prefix}_{str(i + 1).zfill(width)}" for i in range(n)]


def detect_ordinal_candidates(df: pd.DataFrame, max_unique: int = 10) -> list[str]:
    """
    順序尺度候補を簡易判定する
    - 数値型でユニーク数が少ない
    """
    candidates = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= max_unique:
                candidates.append(col)
    return candidates


def coerce_numeric(series: pd.Series) -> pd.Series:
    """
    数値変換できるものは数値へ寄せる
    """
    return pd.to_numeric(series, errors="coerce")


def sample_from_distribution(
    values: np.ndarray,
    probs: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    離散分布からサンプルする
    """
    return rng.choice(values, size=size, p=probs)