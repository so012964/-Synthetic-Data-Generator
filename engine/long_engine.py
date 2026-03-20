from __future__ import annotations

import pandas as pd

from engine.schema import GenerationConfig
from engine.wide_engine import WideModeEngine


class LongModeEngine:
    """
    ロング型向けの簡易生成エンジン

    方針:
    - groupbyキーで群を固定
    - 各群の内部だけ WideModeEngine で生成
    - groupbyキー列は固定して戻す
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.base_engine = WideModeEngine(random_state=random_state)

    def generate(self, df: pd.DataFrame, config: GenerationConfig) -> pd.DataFrame:
        """
        ロング型データを群ごとに生成する
        """
        if not config.groupby_cols:
            raise ValueError("LongModeEngine では groupby_cols が必要です。")

        results = []

        grouped = df.groupby(config.groupby_cols, dropna=False, sort=False)

        for group_keys, group_df in grouped:
            # 群内で生成
            group_config = GenerationConfig(
                mode="wide",
                id_col=None,
                add_row_id=False,
                numeric_cols=config.numeric_cols,
                ordinal_cols=config.ordinal_cols,
                categorical_cols=config.categorical_cols,
                n_rows=len(group_df),
                random_state=config.random_state,
            )

            generated_group = self.base_engine.generate(group_df, group_config)

            # 固定列を戻す
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            for col, val in zip(config.groupby_cols, group_keys):
                generated_group[col] = val

            results.append(generated_group)

        out = pd.concat(results, axis=0, ignore_index=True)

        # 列順をなるべく元データ順に寄せる
        ordered_cols = [c for c in df.columns if c in out.columns]
        out = out[ordered_cols]

        return out