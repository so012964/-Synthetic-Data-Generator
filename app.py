from __future__ import annotations

import io
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine.id_manager import SyntheticIDManager
from engine.profiler import DataProfiler
from engine.schema import GenerationConfig
from engine.utils import make_prefixed_ids
from engine.wide_engine import WideModeEngine

app = FastAPI(title="Synthetic Data Generator")

_file_store: Dict[str, str] = {}
_file_names: Dict[str, str] = {}
_result_store: Dict[str, pd.DataFrame] = {}

MAX_FILTER_OPTIONS = 20
MAX_GENERATION_ROWS = 100_000
_MAX_STORED = 10


def _evict_files() -> None:
    if len(_file_store) > _MAX_STORED:
        for key in list(_file_store.keys())[: len(_file_store) - _MAX_STORED]:
            file_path = _file_store.pop(key, None)
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            _file_names.pop(key, None)


def _evict_results() -> None:
    if len(_result_store) > _MAX_STORED:
        for key in list(_result_store.keys())[: len(_result_store) - _MAX_STORED]:
            _result_store.pop(key, None)


def _format_filter_value(value: Any) -> str:
    if pd.isna(value):
        return "＜欠損値＞"
    return str(value)


def _apply_single_filter(
    df: pd.DataFrame, filter_col: str, selected_values: list[str]
) -> pd.DataFrame:
    if not filter_col or not selected_values:
        return df.copy()
    series = df[filter_col]
    display_map = {_format_filter_value(v): v for v in series.drop_duplicates().tolist()}
    actual_vals = []
    include_na = False
    for val in selected_values:
        if val == "＜欠損値＞":
            include_na = True
        elif val in display_map:
            actual_vals.append(display_map[val])
    mask = series.isin(actual_vals)
    if include_na:
        mask = mask | series.isna()
    return df.loc[mask].copy()


def _compute_stats(
    orig_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    numeric_cols: list[str],
) -> dict:
    cols = [c for c in numeric_cols if c in orig_df.columns and c in syn_df.columns]
    if not cols:
        return {
            "numeric_cols": [], "original_stats": {}, "synthetic_stats": {},
            "original_corr": None, "synthetic_corr": None, "corr_cols": [],
        }
    orig = orig_df[cols]
    syn = syn_df[cols]
    result = {
        "numeric_cols": cols,
        "original_stats": orig.describe().round(4).fillna(0).to_dict(),
        "synthetic_stats": syn.describe().round(4).fillna(0).to_dict(),
        "original_corr": None,
        "synthetic_corr": None,
        "corr_cols": [],
    }
    if len(cols) >= 2:
        result["original_corr"] = orig.corr().round(4).fillna(0).to_dict()
        result["synthetic_corr"] = syn.corr().round(4).fillna(0).to_dict()
        result["corr_cols"] = cols
    return result


def _load_df(file_id: str, sheet_name: str | None = None) -> pd.DataFrame:
    if file_id not in _file_store:
        raise HTTPException(404, "ファイルが見つかりません。再アップロードしてください。")
    file_path = _file_store[file_id]
    file_name = _file_names[file_id].lower()
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "ファイルがサーバーから削除されました。再アップロードしてください。")

    try:
        if file_name.endswith(".csv"):
            try:
                return pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(file_path, encoding="shift-jis")
        return pd.read_excel(file_path, sheet_name=sheet_name or 0)
    except Exception as e:
        raise HTTPException(400, f"ファイルの読み込みに失敗しました: {str(e)}")


@app.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    file_name = file.filename.lower()
    if not (file_name.endswith(".csv") or file_name.endswith((".xlsx", ".xls"))):
        raise HTTPException(400, "CSV または Excel ファイルをアップロードしてください。")

    file_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"syn_gen_{file_id}_{file.filename}")
    
    with open(file_path, "wb") as f:
        f.write(file_bytes)
        
    _file_store[file_id] = file_path
    _file_names[file_id] = file.filename
    _evict_files()

    sheet_names = None
    try:
        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="shift-jis")
        else:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            df = pd.read_excel(file_path, sheet_name=sheet_names[0])
    except Exception as e:
        raise HTTPException(400, f"ファイルの解析に失敗しました。ファイルが破損していないか確認してください: {str(e)}")

    profiler = DataProfiler()
    diagnosis = profiler.diagnose_structure(df)
    profile = profiler.profile_columns(df)

    filter_options: dict = {}
    for col in df.columns:
        if df[col].nunique(dropna=False) <= MAX_FILTER_OPTIONS:
            filter_options[col] = sorted(
                [_format_filter_value(v) for v in df[col].drop_duplicates().tolist()],
                key=str,
            )

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "sheet_names": sheet_names,
        "columns": list(df.columns),
        "row_count": len(df),
        "id_candidates": diagnosis.id_candidates,
        "column_profile": {
            "numeric_cols": profile.numeric_cols,
            "ordinal_cols": profile.ordinal_cols,
            "categorical_cols": profile.categorical_cols,
        },
        "filter_options": filter_options,
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    }


class ProfileRequest(BaseModel):
    file_id: str
    sheet_name: Optional[str] = None
    filter_col: Optional[str] = None
    filter_values: Optional[List[str]] = None
    id_col: Optional[str] = None


@app.post("/api/profile")
def profile_data(req: ProfileRequest):
    df = _load_df(req.file_id, req.sheet_name)
    if req.filter_col and req.filter_values is not None:
        df = _apply_single_filter(df, req.filter_col, req.filter_values)
    if len(df) == 0:
        raise HTTPException(400, "絞り込み後の行数が 0 件です。条件を見直してください。")

    profiler = DataProfiler()
    diagnosis = profiler.diagnose_structure(df)
    profile = profiler.profile_columns(df, id_col=req.id_col)

    return {
        "row_count": len(df),
        "id_candidates": diagnosis.id_candidates,
        "columns": list(df.columns),
        "column_profile": {
            "numeric_cols": profile.numeric_cols,
            "ordinal_cols": profile.ordinal_cols,
            "categorical_cols": profile.categorical_cols,
        },
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    }


class GenerateRequest(BaseModel):
    file_id: str
    sheet_name: Optional[str] = None
    filter_col: Optional[str] = None
    filter_values: Optional[List[str]] = None
    id_col: Optional[str] = None
    add_row_id: bool = False
    numeric_cols: List[str] = []
    ordinal_cols: List[str] = []
    categorical_cols: List[str] = []
    n_rows: Optional[int] = None
    random_state: int = 42


@app.post("/api/generate")
def generate_data(req: GenerateRequest):
    df = _load_df(req.file_id, req.sheet_name)
    if req.filter_col and req.filter_values is not None:
        df = _apply_single_filter(df, req.filter_col, req.filter_values)
    if len(df) == 0:
        raise HTTPException(400, "絞り込み後の行数が 0 件です。条件を見直してください。")
    if req.n_rows is not None and req.n_rows > MAX_GENERATION_ROWS:
        raise HTTPException(400, f"生成行数の上限は {MAX_GENERATION_ROWS:,} 行です。")

    config = GenerationConfig(
        id_col=req.id_col,
        add_row_id=req.add_row_id,
        numeric_cols=req.numeric_cols,
        ordinal_cols=req.ordinal_cols,
        categorical_cols=req.categorical_cols,
        n_rows=req.n_rows,
        random_state=req.random_state,
    )

    df_gen = df.copy()
    if req.id_col is not None:
        id_manager = SyntheticIDManager(prefix="SID")
        df_gen[req.id_col] = id_manager.fit_transform(df_gen[req.id_col])

    engine = WideModeEngine(random_state=req.random_state)
    generated_df = engine.generate(df_gen, config)

    if req.id_col is not None:
        if req.n_rows is None:
            generated_df[req.id_col] = df_gen[req.id_col].values
        else:
            generated_df[req.id_col] = make_prefixed_ids(len(generated_df), prefix="SID")

    if req.add_row_id:
        generated_df.insert(
            0,
            "synthetic_row_id",
            [f"ROW_{str(i + 1).zfill(6)}" for i in range(len(generated_df))],
        )

    stats = _compute_stats(df, generated_df, req.numeric_cols)

    result_id = str(uuid.uuid4())
    _result_store[result_id] = generated_df
    _evict_results()

    return {
        "result_id": result_id,
        "row_count": len(generated_df),
        "columns": list(generated_df.columns),
        "preview": generated_df.head(50).fillna("").to_dict(orient="records"),
        "stats": stats,
    }


@app.get("/api/download/{result_id}")
def download_result(result_id: str, format: str = Query("csv")):
    if result_id not in _result_store:
        raise HTTPException(404, "生成結果が見つかりません。再度生成してください。")
    df = _result_store[result_id]

    if format == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="synthetic_data")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=synthetic_data.xlsx"},
        )

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
