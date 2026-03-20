from __future__ import annotations

import pandas as pd

from engine.utils import normalize_string_series, make_prefixed_ids


class SyntheticIDManager:
    """
    元IDを人工IDへ置換する
    """

    def __init__(self, prefix: str = "SID") -> None:
        self.prefix = prefix
        self.mapping_: dict[str, str] = {}

    def fit(self, series: pd.Series) -> None:
        """
        元ID列から対応表を作る
        """
        normalized = normalize_string_series(series).dropna()
        unique_vals = normalized.drop_duplicates().tolist()

        synthetic_ids = make_prefixed_ids(len(unique_vals), prefix=self.prefix)
        self.mapping_ = dict(zip(unique_vals, synthetic_ids))

    def transform(self, series: pd.Series) -> pd.Series:
        """
        元ID列を人工IDへ変換する
        """
        normalized = normalize_string_series(series)
        return normalized.map(self.mapping_)

    def fit_transform(self, series: pd.Series) -> pd.Series:
        self.fit(series)
        return self.transform(series)