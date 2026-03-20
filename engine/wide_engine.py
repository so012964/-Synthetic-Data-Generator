from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine.schema import GenerationConfig


class WideModeEngine:
    """
    ワイド型向けの簡易生成エンジン

    方針:
    - 数値列: 相関をざっくり保つため、多変量正規で生成
    - 順序列: 元の分布からサンプリング
    - 名義列: 元の分布からサンプリング
    - ID列: 学習対象外
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def _fit_numeric(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict:
        """
        数値列の平均・標準偏差・相関行列を学習する
        """
        if not numeric_cols:
            return {}

        num_df = df[numeric_cols].copy()

        means = num_df.mean()
        stds = num_df.std(ddof=0).replace(0, 1.0)

        z = (num_df - means) / stds
        corr = z.corr().fillna(0.0)

        # 数値安定性のため、対角に小さく足す
        corr_values = corr.values
        corr_values = corr_values + np.eye(len(corr_values)) * 1e-6

        return {
            "means": means,
            "stds": stds,
            "corr": corr_values,
            "numeric_cols": numeric_cols,
        }

    def _generate_numeric(self, fit_result: dict, n_rows: int) -> pd.DataFrame:
        """
        学習した統計量を使って数値列を生成する
        """
        if not fit_result:
            return pd.DataFrame(index=range(n_rows))

        means = fit_result["means"]
        stds = fit_result["stds"]
        corr = fit_result["corr"]
        numeric_cols = fit_result["numeric_cols"]

        # 標準正規ベクトルを相関つきで発生
        z = self.rng.multivariate_normal(
            mean=np.zeros(len(numeric_cols)),
            cov=corr,
            size=n_rows,
        )

        generated = pd.DataFrame(z, columns=numeric_cols)
        generated = generated * stds.values + means.values

        return generated

    def _generate_discrete_from_observed(
        self,
        df: pd.DataFrame,
        cols: list[str],
        n_rows: int,
    ) -> pd.DataFrame:
        """
        順序列・カテゴリ列を元分布ベースで生成する
        """
        out = pd.DataFrame(index=range(n_rows))

        for col in cols:
            vc = df[col].value_counts(dropna=True, normalize=True)
            if len(vc) == 0:
                out[col] = [None] * n_rows
                continue

            values = vc.index.to_numpy()
            probs = vc.values
            out[col] = self.rng.choice(values, size=n_rows, p=probs)

        return out

    def generate(self, df: pd.DataFrame, config: GenerationConfig) -> pd.DataFrame:
        """
        ワイド型データを生成する
        """
        n_rows = config.n_rows if config.n_rows is not None else len(df)

        # -----------------------------
        # 数値列生成
        # -----------------------------
        fit_numeric = self._fit_numeric(df, config.numeric_cols)
        out_num = self._generate_numeric(fit_numeric, n_rows)

        # -----------------------------
        # 順序列・カテゴリ列生成
        # -----------------------------
        out_ord = self._generate_discrete_from_observed(df, config.ordinal_cols, n_rows)
        out_cat = self._generate_discrete_from_observed(df, config.categorical_cols, n_rows)

        # 結合
        out = pd.concat([out_num, out_ord, out_cat], axis=1)

        # 列順をなるべく元データに合わせる
        ordered_cols = [
            col for col in df.columns
            if col in out.columns
        ]
        out = out[ordered_cols]

        return out