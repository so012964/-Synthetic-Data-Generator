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

    # ロング型の繰り返し軸として比較的使われやすいキーワード
    # 「年」「学部」「学科」など広すぎるものは外し、誤判定を減らす
    GROUP_KEYWORDS = [
        "年度",
        "学期",
        "semester",
        "time",
        "時点",
        "科目",
        "設問",
    ]

    def diagnose_structure(self, df: pd.DataFrame) -> DataDiagnosis:
        """
        データ構造を診断して、wide / long / uncertain を推定する
        """
        reasons: list[str] = []
        id_candidates: list[str] = []
        repeated_id_candidates: list[str] = []
        likely_groupby_candidates: list[str] = []

        # -----------------------------
        # ID候補の抽出
        # -----------------------------
        for col in df.columns:
            s = df[col]
            unique_ratio = safe_unique_ratio(s)

            # 列名がIDっぽい、またはユニーク率がかなり高い列を候補にする
            if is_probably_id_name(col) or unique_ratio >= self.ID_UNIQUE_RATIO_THRESHOLD:
                id_candidates.append(col)

        # -----------------------------
        # ID候補のうち、重複を含む列を抽出
        # -----------------------------
        for col in id_candidates:
            # 欠損を除いて重複を見る
            non_na = df[col].dropna()
            if non_na.duplicated().any():
                repeated_id_candidates.append(col)

        # -----------------------------
        # グループ分け候補の抽出
        # -----------------------------
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword.lower() in col_lower for keyword in self.GROUP_KEYWORDS):
                likely_groupby_candidates.append(col)

        # -----------------------------
        # wide / long / uncertain の判定
        # -----------------------------
        has_repeated_id = len(repeated_id_candidates) > 0
        has_group_axis = len(likely_groupby_candidates) > 0

        if has_repeated_id and has_group_axis:
            mode_suggested = "long"
            reasons.append(
                "ID候補列に同じ値が複数回出現しており、"
                "さらに年度・学期・科目・設問のような繰り返し軸になりそうな列もあるため、"
                "ロング型と推定しました。"
            )

        elif has_repeated_id:
            mode_suggested = "uncertain"
            reasons.append(
                "ID候補列に重複が見られますが、"
                "ロング型と断定できるだけの繰り返し軸は十分に確認できなかったため、"
                "判定は保留としました。"
            )

        elif id_candidates:
            mode_suggested = "wide"
            reasons.append(
                "ID候補列の値がほぼ一意であるため、"
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
