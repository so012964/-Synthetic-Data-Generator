from __future__ import annotations

import pandas as pd

from engine.schema import DataDiagnosis, ColumnProfile
from engine.utils import (
    is_probably_id_name,
    safe_unique_ratio,
    detect_ordinal_candidates,
)


class DataProfiler:
    """
    データの性質をざっくり診断するクラス

    判定方針
    ----------
    1. ID候補を抽出する
       - 列名がIDらしい
       - またはユニーク率がかなり高い
    2. ID候補のうち、重複がある列を確認する
    3. ロング型で繰り返し軸になりやすい列を抽出する
    4. それらをもとに wide / long / uncertain を推定する

    注意
    ----------
    以前は「ID候補に重複がある」だけで long に寄りやすかったため、
    今回は long 判定を少し慎重にしている。
    """

    # ID候補とみなすユニーク率のしきい値
    ID_UNIQUE_RATIO_THRESHOLD = 0.95

    def diagnose_structure(self, df: pd.DataFrame) -> DataDiagnosis:
        """
        ID候補列を検出する
        """
        id_candidates: list[str] = []

        for col in df.columns:
            s = df[col]
            unique_ratio = safe_unique_ratio(s)
            if is_probably_id_name(col) or unique_ratio >= self.ID_UNIQUE_RATIO_THRESHOLD:
                id_candidates.append(col)

        return DataDiagnosis(id_candidates=id_candidates)

    def profile_columns(self, df: pd.DataFrame, id_col: str | None = None) -> ColumnProfile:
        """
        各列の型をざっくり分類する

        Parameters
        ----------
        df : pd.DataFrame
            入力データ
        id_col : str | None
            ID列として明示指定された列名

        Returns
        -------
        ColumnProfile
            数値列・順序列・カテゴリ列・ID列の分類結果
        """
        profile = ColumnProfile()

        # 少ないユニーク数を持つ数値列を順序尺度候補として拾う
        ordinal_candidates = set(detect_ordinal_candidates(df))

        for col in df.columns:
            if id_col is not None and col == id_col:
                profile.id_cols.append(col)
                continue

            s = df[col]

            if pd.api.types.is_numeric_dtype(s):
                if col in ordinal_candidates:
                    profile.ordinal_cols.append(col)
                else:
                    profile.numeric_cols.append(col)
            else:
                profile.categorical_cols.append(col)

        return profile
