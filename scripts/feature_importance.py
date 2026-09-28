"""
Pearson Correlation Feature Importance
======================================
Compute per-feature Pearson correlation between embedding dimensions
and a target variable, outputting results as a CSV.

Usage (example)::

    from scripts.feature_importance import pearson_feature_importance
    from scripts.utils import load_parquet

    df = load_parquet(Path("path/to/clap_3.0s.parquet"))
    pearson_feature_importance(
        df,
        target_col="emotion",
        target_values={"anger": 0, "gentleness": 1, "joy": 2, "neutral": 3, "sadness": 4},
        output_path="outputs/feature_importance/emotion_clap_3s.csv",
    )

  # Opensmile: label each dimension with its real feature name from the library
  # (CLAP/Whisper dims have no known meaning, so just omit feature_names for those).
  from scripts.feature_sets.extract_opensmile import opensmile_feature_names

  df = load_parquet(Path("path/to/opensmile_compare-2016_3.0s.parquet"))
  pearson_feature_importance(
      df,
      target_col="emotion",
      target_values={"anger": 0, "gentleness": 1, "joy": 2, "neutral": 3, "sadness": 4},
      feature_names=opensmile_feature_names(),
      output_path="outputs/feature_importance/emotion_opensmile_3s.csv",
  )
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _encode_target(
    df: pd.DataFrame,
    target_col: str,
    target_values: Optional[dict[str, float]] = None,
) -> tuple[pd.Series, dict[float, str]]:
    """
    Encode the target column to numeric.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Column name in ``df``.
    target_values : dict, optional
        Mapping ``{label: numeric_value}`` for non-numeric targets.
        Required when the column contains strings.

    Returns
    -------
    encoded : pd.Series
    reverse_mapping : dict
        Maps encoded value back to original label (empty when target is
        already numeric).
    """
    series = df[target_col]

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(np.float64), {}

    if target_values is None:
        raise ValueError(
            f"Column '{target_col}' is non-numeric. Provide a "
            f"target_values mapping dict, e.g. "
            f"{{'anger': 0, 'joy': 1, ...}}."
        )

    unique_cats = set(series.dropna().unique())
    missing = unique_cats - set(target_values.keys())
    if missing:
        raise ValueError(
            f"target_values is missing categories found in '{target_col}': "
            f"{sorted(missing)}"
        )

    encoded = series.map(target_values).astype(np.float64)
    reverse = {v: k for k, v in target_values.items()}
    return encoded, reverse


def _expand_embeddings(
    df: pd.DataFrame,
    embedding_col: str = "embedding",
) -> pd.DataFrame:
    """
    Unpack the array-valued ``embedding_col`` into one column per
    dimension, named ``{embedding_col}_0``, ``{embedding_col}_1``, etc.
    """
    embeddings = np.stack(df[embedding_col].to_numpy())
    cols = [f"{embedding_col}_{i}" for i in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=cols, index=df.index)


def pearson_feature_importance(
    df: pd.DataFrame,
    target_col: str,
    target_values: Optional[dict[str, float]] = None,
    embedding_col: str = "embedding",
    feature_cols: Optional[list[str]] = None,
    feature_names: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Compute Pearson correlation of each feature with a target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.  Must contain the target column and either an
        ``embedding_col`` (numpy arrays) or the columns listed in
        ``feature_cols``.
    target_col : str
        Column to predict / correlate against.
    target_values : dict, optional
        ``{label: numeric}`` mapping for categorical targets.  Ignored
        when ``target_col`` is already numeric.
    embedding_col : str
        Column containing per-row embedding vectors (numpy arrays).
        Ignored when ``feature_cols`` is given.
    feature_cols : list of str, optional
        Use these pre-expanded feature columns directly (e.g. opensmile
        column names).  When ``None``, ``embedding_col`` is expanded.
    feature_names : list of str, optional
        Human-readable names for each embedding dimension, in order.
        Usage: pass ``opensmile_feature_names()`` for opensmile so results
        are labelled like 'pcm_loudness_sma3_amean' instead of
        'embedding_0'.  Ignored when ``feature_cols`` is given (those
        column names are used directly).  For CLAP/Whisper there is no
        known mapping, so omit this and the ``{embedding_col}_N`` names
        are kept.
    output_path : str or Path, optional
        If given the results CSV is written here (parent dirs created
        automatically).

    Returns
    -------
    pd.DataFrame
        Columns ``[feature, pearson_r]``, sorted by descending
        ``|pearson_r|``.
    """
    df_work = df.copy()

    # --- feature matrix ---
    if feature_cols is not None:
        feature_names_final = list(feature_cols)
        feature_df = df_work[feature_cols].astype(np.float64)
    else:
        feature_df = _expand_embeddings(df_work, embedding_col)
        if feature_names is not None:
            if len(feature_names) != feature_df.shape[1]:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries but "
                    f"'{embedding_col}' expands to {feature_df.shape[1]} "
                    f"dimensions."
                )
            feature_names_final = list(feature_names)
        else:
            feature_names_final = list(feature_df.columns)

    # --- target ---
    encoded_target, _reverse = _encode_target(df_work, target_col, target_values)

    # --- correlation ---
    valid_mask = encoded_target.notna()
    results: list[tuple[str, float]] = []

    for idx, feat in enumerate(feature_df.columns):
        feat_series = feature_df[feat]
        mask = valid_mask & feat_series.notna()
        n = mask.sum()
        if n < 2:
            results.append((feature_names_final[idx], 0.0))
            continue
        r = feat_series.loc[mask].corr(encoded_target.loc[mask], method="pearson")
        results.append((feature_names_final[idx], r if np.isfinite(r) else 0.0))

    result = (
        pd.DataFrame(results, columns=["feature", "pearson_r"])
        .assign(_abs=lambda c: c["pearson_r"].abs())
        .sort_values("_abs", ascending=False, kind="stable")
        .drop(columns=["_abs"])
        .reset_index(drop=True)
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        print(f"[INFO] Feature importance saved to {out}")

    return result


def select_top_features(
    df: pd.DataFrame,
    target_col: str,
    n_top: int = 100,
    target_values: Optional[dict[str, float]] = None,
    embedding_col: str = "embedding",
    feature_cols: Optional[list[str]] = None,
    feature_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select the top ``n_top`` features for a target, ranked by |Pearson r|.

    Convenience wrapper used by feature_visualisation to build
    cross-embedding correlation heatmaps.

    Parameters
    ----------
    df, target_col, target_values, embedding_col, feature_cols, feature_names
        Same meanings as :func:`pearson_feature_importance`.

    Returns
    -------
    (feature_matrix, display_names)
        feature_matrix : the original feature columns restricted to the
            top-n features (internal column names, same row order as df).
        display_names  : human-readable label per selected column, in the
            same order (uses ``feature_names`` when given).
    """
    imp = pearson_feature_importance(
        df,
        target_col,
        target_values,
        embedding_col,
        feature_cols,
        feature_names,
    )
    top_display = imp.head(n_top)["feature"].tolist()
    if not top_display:
        raise ValueError(f"No features available for target column '{target_col}'.")

    if feature_cols is not None:
        feature_df = df[feature_cols].astype(np.float64)
        internal_cols = top_display
    else:
        feature_df = _expand_embeddings(df, embedding_col)
        if feature_names is None:
            internal_cols = top_display
        else:
            index_map = {display: i for i, display in enumerate(feature_names)}
            internal_cols = [
                f"{embedding_col}_{index_map[display]}" for display in top_display
            ]

    return feature_df.loc[df.index, internal_cols], top_display


# ---------------------------------------------------------------------------
# Demo: run directly for the TUNI CLAP 3.0s case
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scripts.utils import load_parquet

    PARQUET = Path(__file__).resolve().parent.parent / "TUNI_emotion_data" / "embeddings" / "tuni_emotion_dataset_clap-2023_3.0s.parquet"
    OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "feature_importance" / "tuni_clap_3s_emotion.csv"

    if not PARQUET.exists():
        # Fallback: search the default output dir used by extraction configs
        PARQUET = Path(
            "/home/suvihaara/Documents/PhD/DATA/TUNI_emotion_dataset/embeddings/"
            "tuni_emotion_dataset_clap-2023_3.0s.parquet"
        )

    df = load_parquet(PARQUET)
    print(f"Loaded {len(df)} rows from {PARQUET}")

    result = pearson_feature_importance(
        df,
        target_col="emotion",
        target_values={
            "anger": 0,
            "gentleness": 1,
            "joy": 2,
            "neutral": 3,
            "sadness": 4,
        },
        output_path=OUTPUT,
    )

    print("\nTop 100 features by |Pearson r|:")
    print(result.head(100).to_string(index=False))
