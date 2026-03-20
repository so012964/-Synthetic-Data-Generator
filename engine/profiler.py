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
    """

    def diagnose_structure(self, df: pd.DataFrame) -> DataDiagnosis:
        reasons: list[str] = []
        id_candidates: list[str] = []
        repeated_id_candidates: list[str] = []
        likely_groupby_candidates: list[str] = []

        # -----------------------------
        # ID候補の抽出
        # 列名がID的、またはユニーク率が高い列を候補にする
        # -----------------------------
        for col in df.columns:
            s = df[col]
            unique_ratio = safe_unique_ratio(s)

            if is_probably_id_name(col) or unique_ratio > 0.8:
                id_candidates.append(col)

        # -----------------------------
        # ID重複の確認
        # -----------------------------
        for col in id_candidates:
            if df[col].duplicated().any():
                repeated_id_candidates.append(col)

        # -----------------------------
        # グループ分け候補の抽出
        # ロング型で繰り返しの軸になりやすい列を候補にする
        # -----------------------------
        group_keywords = ["年度", "年", "学期", "semester", "time", "時点", "科目", "項目", "設問", "学部", "学科"]
        for col in df.columns:
            if any(k.lower() in str(col).lower() for k in group_keywords):
                likely_groupby_candidates.append(col)

        # -----------------------------
        # ワイド型 / ロング型の簡易推定
        # -----------------------------
        if repeated_id_candidates and likely_groupby_candidates:
            mode_suggested = "long"
            reasons.append(
                "ID候補列に同じ値が複数回出現しており、"
                "年度や科目のような繰り返しの軸になりそうな列もあるため、"
                "ロング型と推定しました。"
            )
        elif repeated_id_candidates:
            mode_suggested = "long"
            reasons.append(
                "ID候補列に同じ値が複数回出現しているため、"
                "ロング型と推定しました。"
            )
        elif id_candidates:
            mode_suggested = "wide"
            reasons.append(
                "ID候補列の値がほぼすべて異なるため、"
                "ワイド型と推定しました。"
            )
        else:
            mode_suggested = "uncertain"
            reasons.append(
                "明確なID候補列が見つからなかったため、"
                "データ形式の判定は保留としました。"
            )

        return DataDiagnosis(
            mode_suggested=mode_suggested,
            id_candidates=id_candidates,
            repeated_id_candidates=repeated_id_candidates,
            likely_groupby_candidates=likely_groupby_candidates,
            reasons=reasons,
        )

    def profile_columns(self, df: pd.DataFrame, id_col: str | None = None) -> ColumnProfile:
        """
        各列の型をざっくり分類する
        """
        profile = ColumnProfile()

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