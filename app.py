from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from engine.profiler import DataProfiler
from engine.id_manager import SyntheticIDManager
from engine.schema import GenerationConfig, MIN_GROUP_SIZE
from engine.wide_engine import WideModeEngine
from engine.long_engine import LongModeEngine


st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧪",
    layout="wide",
)

# -----------------------------------------------
# Zen風カスタムCSS
# -----------------------------------------------
st.markdown(
    """
    <style>
    /* --- 見出し --- */
    h1 {
        color: #2D3436;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-bottom: 2px solid #6B9080;
        padding-bottom: 0.4rem;
    }
    h2, h3 {
        color: #3D5A4C;
        font-weight: 500;
    }

    /* --- ボタン --- */
    .stButton > button[kind="primary"] {
        background-color: #6B9080;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #5A7A6C;
    }
    .stButton > button {
        border-radius: 6px;
    }

    /* --- ダウンロードボタン --- */
    .stDownloadButton > button {
        background-color: #F0EDE8;
        color: #2D3436;
        border: 1px solid #D5CEC5;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stDownloadButton > button:hover {
        background-color: #E5E0D9;
    }

    /* --- データフレーム --- */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* --- expander --- */
    .streamlit-expanderHeader {
        font-size: 0.95rem;
        color: #3D5A4C;
        font-weight: 500;
    }

    /* --- 区切り線 --- */
    hr {
        border: none;
        border-top: 1px solid #D5CEC5;
        margin: 1.5rem 0;
    }

    /* --- フッターを非表示 --- */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# フィルタ列で一覧表示できる値の上限
MAX_FILTER_OPTIONS = 20

# 生成行数の上限
MAX_GENERATION_ROWS = 100_000


@st.cache_data(ttl=600)
def get_excel_sheet_names(uploaded_file_bytes: bytes) -> list[str]:
    """Excelファイルのシート名一覧を取得する"""
    excel_file = pd.ExcelFile(io.BytesIO(uploaded_file_bytes))
    return excel_file.sheet_names


@st.cache_data(ttl=600)
def load_csv_file(uploaded_file_bytes: bytes) -> pd.DataFrame:
    """CSVファイルを読み込む"""
    return pd.read_csv(io.BytesIO(uploaded_file_bytes))


@st.cache_data(ttl=600)
def load_excel_file(uploaded_file_bytes: bytes, sheet_name: str | int = 0) -> pd.DataFrame:
    """Excelファイルを指定シートで読み込む"""
    return pd.read_excel(io.BytesIO(uploaded_file_bytes), sheet_name=sheet_name)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """DataFrameをExcelバイト列へ変換する"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="synthetic_data")
    return buffer.getvalue()


def format_filter_value(value) -> str:
    """フィルタの候補値を画面表示用の文字列に変換する"""
    if pd.isna(value):
        return "＜欠損値＞"
    return str(value)


def apply_single_filter(df: pd.DataFrame, filter_col: str, selected_values: list[str]) -> pd.DataFrame:
    """
    1つの列に対するフィルタを適用する

    Parameters
    ----------
    df : pd.DataFrame
        元のデータフレーム
    filter_col : str
        フィルタ対象の列名
    selected_values : list[str]
        画面上で選択された値（表示用文字列）のリスト

    Returns
    -------
    pd.DataFrame
        フィルタ適用後のデータフレーム
    """
    if not filter_col or not selected_values:
        return df.copy()

    series = df[filter_col]

    # 元の値と表示用文字列の対応表を作る
    display_map = {}
    for v in series.drop_duplicates().tolist():
        display_map[format_filter_value(v)] = v

    selected_actual_values = []
    include_na = False

    for val in selected_values:
        if val == "＜欠損値＞":
            include_na = True
        elif val in display_map:
            selected_actual_values.append(display_map[val])

    mask = series.isin(selected_actual_values)

    if include_na:
        mask = mask | series.isna()

    return df.loc[mask].copy()


def compute_group_size_stats(
    df: pd.DataFrame, groupby_cols: list[str]
) -> dict:
    """
    グループ分け列ごとの行数統計を返す

    Returns
    -------
    dict
        total_groups : グループ数
        min_size     : 最小行数
        median_size  : 中央値
        max_size     : 最大行数
        small_groups : MIN_GROUP_SIZE 未満のグループ数
        size_series  : グループごとの行数 Series
    """
    group_sizes = df.groupby(groupby_cols, dropna=False, sort=False).size()
    return {
        "total_groups": len(group_sizes),
        "min_size": int(group_sizes.min()),
        "median_size": float(group_sizes.median()),
        "max_size": int(group_sizes.max()),
        "small_groups": int((group_sizes < MIN_GROUP_SIZE).sum()),
        "size_series": group_sizes,
    }


def main() -> None:
    st.title("Synthetic Data Generator")
    st.caption(
        "元データの構造や分布を参考にしながら、"
        "分析パイプラインの動作確認に使えるデータを生成します。"
    )

    st.info(
        "CSV はそのまま読み込みます。"
        "Excel はシートが複数ある場合に対象シートを選択できます。"
    )

    uploaded_file = st.file_uploader(
        "CSV または Excel ファイルを選択してください",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is None:
        st.info("ファイルをアップロードしてください。")
        return

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    # -----------------------------
    # データ読み込み
    # -----------------------------
    try:
        if file_name.endswith(".csv"):
            raw_df = load_csv_file(file_bytes)

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            sheet_names = get_excel_sheet_names(file_bytes)

            if len(sheet_names) == 1:
                selected_sheet = sheet_names[0]
                st.write(f"**対象シート:** {selected_sheet}（シートが1枚のため自動選択）")
            else:
                selected_sheet = st.selectbox(
                    "対象シートを選択してください",
                    options=sheet_names,
                    index=0,
                )
                st.caption("変更しない場合は先頭のシートを使用します。")

            raw_df = load_excel_file(file_bytes, sheet_name=selected_sheet)

        else:
            st.error("CSV または Excel ファイルをアップロードしてください。")
            return

    except Exception as e:
        st.error(f"読み込みに失敗しました: {e}")
        return

    st.subheader("アップロードデータのプレビュー")
    st.dataframe(raw_df.head(), use_container_width=True)

    # -----------------------------
    # アプリ内フィルタ
    # -----------------------------
    st.subheader("データの絞り込み")
    st.caption(
        "必要に応じて、生成前にデータを絞り込めます。"
        "使わない場合はそのまま進めてください。"
    )

    use_filter = st.checkbox("絞り込みを使う", value=False)

    working_input_df = raw_df.copy()

    if use_filter:
        filter_col = st.selectbox(
            "絞り込みに使う列",
            options=list(raw_df.columns),
        )

        unique_count = raw_df[filter_col].nunique(dropna=False)
        st.write(f"**この列の異なる値の数:** {unique_count}")

        if unique_count <= MAX_FILTER_OPTIONS:
            filter_candidates = [
                format_filter_value(v)
                for v in raw_df[filter_col].drop_duplicates().tolist()
            ]

            # 表示順を安定させるため文字列順で並べる
            filter_candidates = sorted(filter_candidates, key=str)

            selected_filter_values = st.multiselect(
                "残す値",
                options=filter_candidates,
                default=filter_candidates,
            )

            working_input_df = apply_single_filter(
                df=raw_df,
                filter_col=filter_col,
                selected_values=selected_filter_values,
            )

            st.write(f"**絞り込み後の行数:** {len(working_input_df):,} / {len(raw_df):,}")

            if len(working_input_df) == 0:
                st.warning("絞り込み後の行数が 0 件です。条件を見直してください。")
                return

        else:
            st.warning(
                f"この列は異なる値が {MAX_FILTER_OPTIONS} 件を超えているため、"
                f"一覧からは選択できません。"
            )
            st.caption(
                "異なる値の少ない列を選ぶか、"
                "事前にデータを加工してからご利用ください。"
            )
            working_input_df = raw_df.copy()

    st.subheader("生成対象データのプレビュー")
    st.dataframe(working_input_df.head(), use_container_width=True)

    profiler = DataProfiler()
    diagnosis = profiler.diagnose_structure(working_input_df)

    st.subheader("データ診断")
    st.write(f"**推定モード:** {diagnosis.mode_suggested}")
    for reason in diagnosis.reasons:
        st.write(f"- {reason}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**ID候補列**")
        st.write(diagnosis.id_candidates if diagnosis.id_candidates else "候補なし")

        st.write("**重複のあるID候補**")
        st.write(diagnosis.repeated_id_candidates if diagnosis.repeated_id_candidates else "なし")

    with col2:
        st.write("**グループ分け候補列**")
        st.write(diagnosis.likely_groupby_candidates if diagnosis.likely_groupby_candidates else "候補なし")

    # -----------------------------
    # 基本設定
    # -----------------------------
    st.subheader("基本設定")

    mode = st.radio(
        "処理モード",
        options=["wide", "long"],
        index=0 if diagnosis.mode_suggested == "wide" else 1 if diagnosis.mode_suggested == "long" else 0,
        horizontal=True,
    )

    id_col_options = ["なし"] + list(working_input_df.columns)
    default_id_idx = 0
    if diagnosis.id_candidates:
        default_id_idx = id_col_options.index(diagnosis.id_candidates[0])

    id_col_raw = st.selectbox(
        "ID列（学籍番号など）",
        options=id_col_options,
        index=default_id_idx,
        help="個人や行を識別するための列です。生成時に人工IDへ自動で置き換えられます。",
    )
    id_col = None if id_col_raw == "なし" else id_col_raw

    add_row_id = st.checkbox(
        "各行に通し番号をつける",
        value=(mode == "long"),
        help=(
            "生成データの先頭列に ROW_000001, ROW_000002, ... "
            "のような通し番号の列を追加します。\n\n"
            "生成後のデータには元データの行番号が残らないため、"
            "「何行目のデータか」を後から確認したいときに便利です。\n\n"
            "特にロング型データでは、同じIDの人が複数行にまたがるため、"
            "行単位で特定したい場合にはオンにしておくのがおすすめです。"
        ),
    )

    random_state = st.number_input("乱数シード", min_value=0, value=42, step=1)

    # -----------------------------
    # 列の性質
    # -----------------------------
    st.subheader("列の性質")

    profile = profiler.profile_columns(working_input_df, id_col=id_col)

    st.caption("各列の型を自動で判定しました。必要に応じて修正してください。")

    numeric_cols = st.multiselect(
        "数値列",
        options=[c for c in working_input_df.columns if c != id_col],
        default=profile.numeric_cols,
    )

    ordinal_cols = st.multiselect(
        "順序列（5段階評価など）",
        options=[c for c in working_input_df.columns if c != id_col and c not in numeric_cols],
        default=[c for c in profile.ordinal_cols if c not in numeric_cols],
    )

    categorical_cols = st.multiselect(
        "カテゴリ列（学部・性別など）",
        options=[c for c in working_input_df.columns if c != id_col and c not in numeric_cols and c not in ordinal_cols],
        default=[c for c in profile.categorical_cols if c not in numeric_cols and c not in ordinal_cols],
    )

    # -----------------------------
    # ロング型向け設定
    # -----------------------------
    groupby_cols: list[str] = []
    if mode == "long":
        st.subheader("グループ分けして生成（ロング型向け）")

        # --- 意義の説明 ---
        with st.expander("なぜグループ分けが必要なのか", expanded=False):
            st.markdown(
                """
**グループ分けの目的**

ロング型データには「年度ごと」「科目ごと」のように、
グループによって値の傾向が異なるケースがよくあります。
グループ分けをせずに全体から一括で生成すると、
こうした**グループごとの違いが平均化されて消えてしまいます**。

例えば、数学の平均点が80点・英語の平均点が50点だった場合、
一括で生成すると全体の平均65点付近にまとまったデータになります。
科目ごとに分けて生成すれば、数学は80点付近・英語は50点付近という傾向が保たれます。

カテゴリ列ではさらに影響が大きく、
ある科目では工学部の学生しかいなかったのに、
一括生成では文学部の学生が混ざるといった
**元データに存在しなかった組み合わせ**が生まれることがあります。

---

**選び方の目安**

「この列の値が変わると、他の数値の傾向も変わりそう」と思う列を選んでください。
迷ったら、**集計やグラフを作るときに横軸や分類軸に使いそうな列**を
選ぶのがおすすめです。

---

**注意点**

列を増やすほどグループが細かくなり、1グループあたりのデータ件数が減ります。
件数が少なすぎると元の傾向をうまく反映できなくなるため、
**まずは1〜2列から**試してみてください。

1グループあたり**{min_size}行以上**あることが望ましいです。
これを下回る場合は警告を表示します。
""".format(min_size=MIN_GROUP_SIZE)
            )

        # --- 列選択 ---
        groupby_candidates = [c for c in working_input_df.columns if c != id_col]

        groupby_cols = st.multiselect(
            "グループ分けに使う列",
            options=groupby_candidates,
            default=[
                c for c in diagnosis.likely_groupby_candidates
                if c in working_input_df.columns and c != id_col
            ][:2],
        )

        # --- グループサイズの検証 ---
        if groupby_cols:
            stats = compute_group_size_stats(working_input_df, groupby_cols)

            st.markdown(
                f"**グループ数:** {stats['total_groups']:,}　／　"
                f"**最小行数:** {stats['min_size']:,}　／　"
                f"**中央値:** {stats['median_size']:.0f}　／　"
                f"**最大行数:** {stats['max_size']:,}"
            )

            if stats["small_groups"] > 0:
                st.warning(
                    f"1グループあたりの行数が {MIN_GROUP_SIZE} 行未満のグループが "
                    f"**{stats['small_groups']}件** あります。\n\n"
                    f"行数が少なすぎるグループでは、元データのたまたまの偏りが "
                    f"生成データに増幅されるため、分布の再現精度が下がります。\n\n"
                    f"グループ分けの列を減らすか、事前にデータを絞り込んでから "
                    f"お試しください。このまま生成することもできます。"
                )

    # -----------------------------
    # 生成行数
    # -----------------------------
    st.subheader("生成行数")

    if mode == "long" and groupby_cols:
        st.caption(
            "ロング型でグループ分けを使用する場合、"
            "グループ構造を維持するため行数は元データと同じになります。"
        )
        n_rows = None
    else:
        n_rows_mode = st.radio(
            "生成行数",
            options=["元データと同じ", "指定する"],
            horizontal=True,
        )
        if n_rows_mode == "元データと同じ":
            n_rows = None
        else:
            n_rows = st.number_input(
                "生成行数",
                min_value=1,
                max_value=MAX_GENERATION_ROWS,
                value=min(len(working_input_df), MAX_GENERATION_ROWS),
                step=1,
                help=f"サーバー負荷を考慮し、上限は {MAX_GENERATION_ROWS:,} 行です。",
            )

    st.write(f"**生成予定行数:** {n_rows if n_rows is not None else len(working_input_df):,} 行")

    # -----------------------------
    # 実行
    # -----------------------------
    st.markdown("---")

    if st.button("Generate", type="primary"):
        try:
            config = GenerationConfig(
                mode=mode,
                id_col=id_col,
                add_row_id=add_row_id,
                groupby_cols=groupby_cols,
                numeric_cols=numeric_cols,
                ordinal_cols=ordinal_cols,
                categorical_cols=categorical_cols,
                n_rows=n_rows,
                random_state=int(random_state),
            )

            df_for_generation = working_input_df.copy()

            # -----------------------------
            # ID列を人工IDへ置換（生成エンジンの学習対象には含めない）
            # -----------------------------
            if id_col is not None:
                id_manager = SyntheticIDManager(prefix="SID")
                df_for_generation[id_col] = id_manager.fit_transform(df_for_generation[id_col])

            # -----------------------------
            # エンジン分岐
            # -----------------------------
            if mode == "wide":
                engine = WideModeEngine(random_state=int(random_state))
                generated_df = engine.generate(df_for_generation, config)

                # 生成後にID列を補填
                if id_col is not None:
                    if n_rows is None:
                        generated_df[id_col] = df_for_generation[id_col].values
                    else:
                        generated_df[id_col] = [f"SID_{str(i + 1).zfill(6)}" for i in range(len(generated_df))]

            else:
                if not groupby_cols:
                    st.error("ロング型では少なくとも1つ「グループ分けに使う列」を指定してください。")
                    return

                engine = LongModeEngine(random_state=int(random_state))
                generated_df = engine.generate(df_for_generation, config)

                # 生成後にID列を補填
                if id_col is not None:
                    if n_rows is None:
                        generated_df[id_col] = df_for_generation[id_col].values
                    else:
                        generated_df[id_col] = [f"SID_{str(i + 1).zfill(6)}" for i in range(len(generated_df))]

            # -----------------------------
            # 通し番号の追加
            # -----------------------------
            if add_row_id:
                generated_df.insert(
                    0,
                    "synthetic_row_id",
                    [f"ROW_{str(i + 1).zfill(6)}" for i in range(len(generated_df))]
                )

            st.success("データを生成しました。")

            st.subheader("生成結果のプレビュー")
            st.dataframe(generated_df.head(50), use_container_width=True)

            csv_bytes = generated_df.to_csv(index=False).encode("utf-8-sig")
            xlsx_bytes = to_excel_bytes(generated_df)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="CSV をダウンロード",
                    data=csv_bytes,
                    file_name="synthetic_data.csv",
                    mime="text/csv",
                )
            with col_dl2:
                st.download_button(
                    label="Excel をダウンロード",
                    data=xlsx_bytes,
                    file_name="synthetic_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"生成中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
