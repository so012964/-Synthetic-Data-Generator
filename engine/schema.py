from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# グループあたりの最小行数（これを下回ると警告を表示する）
MIN_GROUP_SIZE = 15


@dataclass
class DataDiagnosis:
    """データ診断結果を保持するクラス"""

    mode_suggested: str  # "wide", "long", "uncertain"
    id_candidates: List[str] = field(default_factory=list)
    repeated_id_candidates: List[str] = field(default_factory=list)
    likely_groupby_candidates: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class ColumnProfile:
    """各列の型情報"""

    numeric_cols: List[str] = field(default_factory=list)
    ordinal_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    id_cols: List[str] = field(default_factory=list)
    excluded_cols: List[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """生成設定"""

    mode: str  # "wide" or "long"
    id_col: Optional[str]
    add_row_id: bool
    row_id_col: str = "synthetic_row_id"
    groupby_cols: List[str] = field(default_factory=list)
    numeric_cols: List[str] = field(default_factory=list)
    ordinal_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    n_rows: Optional[int] = None  # Noneなら元データ行数
    random_state: int = 42