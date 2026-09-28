import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
<<<<<<< Updated upstream
from matplotlib.colors import LinearSegmentedColormap
=======

from scripts.feature_importance import select_top_features

>>>>>>> Stashed changes

def prepare_and_plot_lda(
    df,
    embedding_col="embedding",
    class_col="singer",  # Column to use for LDA clustering
    mode="2d",  # "2d" or "3d"
    palette="viridis",
    data_type="Embedding Features",
    filters=None,
    plot_label_col=None,  # New: Column to use for coloring the plot, defaults to class_col
):
    """
    Prepares embeddings from a dataframe and visualises using LDA.

    Parameters:
    - df: pandas DataFrame containing embeddings and metadata
    - embedding_col: name of the column containing embeddings (as lists)
    - class_col: column to use as class labels for LDA dimensionality reduction
    - mode: "2d" or "3d" for LDA projection
    - palette: seaborn colour palette
    - data_type: descriptive name for title
    - filters: dict of {column_name: list_of_values} to filter dataframe
    - plot_label_col: column to use for coloring the scatter plot. If None, uses class_col.
    """

    # Copy df to avoid modifying original
    data = df.copy()

    # Apply filters if any
    if filters:
        for col, vals in filters.items():
            data = data[data[col].isin(vals)]

    if data.shape[0] == 0:
        print("Warning: No rows left after filtering. Nothing to plot.")
        return

    # Flatten embeddings
    X = np.array(data[embedding_col].tolist())

    # 'y_lda' is used for the LDA fitting, based on class_col
    y_lda = data[class_col].values
    n_classes = len(np.unique(y_lda))

    if n_classes < 2:
        print(
            f"Warning: Only one class ({y_lda[0]}) after filtering. LDA cannot be applied."
        )
        return

    # LDA: determine number of components based on data
    n_components = min(3, n_classes - 1)
    lda = LDA(n_components=n_components)
    embedding = lda.fit_transform(X, y_lda)

    # Determine the column to use for plotting hue/labels
    plot_hue_source = plot_label_col if plot_label_col is not None else class_col
    if plot_hue_source not in data.columns:
        print(f"Error: Plot label column '{plot_hue_source}' not found in DataFrame.")
        return

    # Prepare data for plotting
    data_plot = data.copy()
    data_plot["LD1"] = embedding[:, 0]
    data_plot["LD2"] = (
        embedding[:, 1] if n_components >= 2 else np.zeros_like(embedding[:, 0])
    )
    data_plot["LD3"] = (
        embedding[:, 2] if n_components >= 3 else np.zeros_like(embedding[:, 0])
    )
    data_plot["Plot_Hue_Label"] = data[plot_hue_source].values

    # Resolve palette: support special 'musa_palette' which maps classes to
    # the project's custom colours.
    if isinstance(palette, str) and palette == "musa_palette":
        hue_names = list(data_plot["Plot_Hue_Label"].unique())
        n_labels = len(hue_names)

        # Cyan → purple → pink
        musa_cmap = LinearSegmentedColormap.from_list(
            "musa",
            ["#6afcfa", "#8A5ACD", "#fe73ab"]
        )

        # Generate exactly as many colours as there are labels
        colours = [musa_cmap(x) for x in np.linspace(0, 1, n_labels)]

        # Convert RGBA → hex
        colours = [
            "#{:02x}{:02x}{:02x}".format(
                int(r * 255),
                int(g * 255),
                int(b * 255)
            )
            for r, g, b, _ in colours
        ]

        colour_map = {
            name: colour
            for name, colour in zip(hue_names, colours)
        }

        resolved_palette = colour_map

    else:
        resolved_palette = palette

    # Plot
    if mode == "2d" or n_components < 3:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="LD1", y="LD2", hue="Plot_Hue_Label", data=data_plot, palette=resolved_palette
        )
        plt.title(f"LDA Projection of {data_type} (2D)")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.show()
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        hue_class_names = list(data_plot["Plot_Hue_Label"].unique())
        if isinstance(palette, str) and palette == "musa_palette":
            colours = [resolved_palette[name] for name in hue_class_names]
        else:
            colours = sns.color_palette(palette, n_colors=len(hue_class_names))

        for i, hue_cls in enumerate(hue_class_names):
            idx = data_plot["Plot_Hue_Label"] == hue_cls
            ax.scatter(
                data_plot.loc[idx, "LD1"],
                data_plot.loc[idx, "LD2"],
                data_plot.loc[idx, "LD3"],
                color=colours[i],
                alpha=0.7,
                label=hue_cls,
            )
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_zlabel("LD3")
        ax.set_title(f"LDA Projection of {data_type} (3D)")
        ax.legend(title="Classes", loc="best")
        plt.show()


def _thinned_tick_labels(
    labels: list[str],
    max_ticks: int = 20,
) -> list[str]:
    """
    Downsample tick labels for wide heatmaps so they stay readable.
    Keeps roughly ``max_ticks`` labels spread evenly, blanking the rest.
    """
    n = len(labels)
    if n <= max_ticks:
        return labels
    step = int(np.ceil(n / max_ticks))
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]


def _frame_keys(df: pd.DataFrame) -> Optional[pd.Series]:
    """Key per row identifying the source frame, if filepath/frame_index exist."""
    if {"filepath", "frame_index"}.issubset(df.columns):
        return df["filepath"].astype(str) + "|" + df["frame_index"].astype(str)
    return None


def correlation_heatmaps_between_embeddings(
    dfs: dict[str, pd.DataFrame],
    target_col: str,
    target_values: Optional[dict[str, float]] = None,
    n_features: int = 100,
    feature_names: Optional[dict[str, Optional[list[str]]]] = None,
    embedding_col: str = "embedding",
    colormap: str = "vlag",
    save_dir: Optional[str | Path] = None,
    show: bool = True,
) -> dict[tuple[str, str], np.ndarray]:
    """
    Cross-correlation heatmaps between embedding types.

    For every pair of embeddings the ``n_features`` most relevant features
    of each (ranked by |Pearson r| with ``target_col``) are selected, the
    two sets are aligned on shared frames, and their pairwise Pearson
    correlations are drawn as a heatmap — e.g. clap-opensmile,
    whisper-opensmile, clap-whisper.

    Opensmile features automatically get their real library names
    (e.g. 'pcm_loudness_sma3_amean'); CLAP/Whisper dimensions have no known
    meaning and stay labelled ``embedding_0``, ``embedding_1``, ...  Pass
    ``feature_names`` to override per embedding.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping embedding name -> dataframe (loaded via scripts.utils
        ``load_parquet``).  Keys become heatmap axis titles.
    target_col : str
        Column the top features are ranked by (e.g. 'emotion').
    target_values : dict, optional
        ``{label: numeric}`` mapping for categorical targets (see
        pearson_feature_importance).
    n_features : int
        How many top features per embedding to include (default 100, so
        the heatmaps stay readable).
    feature_names : dict[str, list[str]], optional
        Optional display-name overrides per embedding name.
    embedding_col : str
        Name of the embedding column in every dataframe.
    colormap : str
        Seaborn/matplotlib colormap for the heatmaps.
    save_dir : str or Path, optional
        If given each heatmap is saved here as PNG.
    show : bool
        Whether to display the heatmaps interactively.

    Returns
    -------
    dict[(name_a, name_b), np.ndarray]
        Pair -> correlation matrix (rows = name_b features, cols = name_a
        features).
    """
    names = list(dfs.keys())
    if len(names) < 2:
        raise ValueError("Provide at least two embedding dataframes in `dfs`.")

    def display_names_for(name: str) -> Optional[list[str]]:
        if feature_names is not None and name in feature_names:
            return feature_names[name]
        if "opensmile" in name.lower():
            # Lazy import so opensmile is only loaded when actually needed.
            from scripts.feature_sets.extract_opensmile import opensmile_feature_names

            return opensmile_feature_names()
        return None

    # --- select top features per embedding and remember shared-row keys ---
    matrices: dict[str, pd.DataFrame] = {}
    disp: dict[str, list[str]] = {}
    has_keys: dict[str, bool] = {}
    for name in names:
        df = dfs[name]
        keys = _frame_keys(df)
        has_keys[name] = keys is not None
        mat, display = select_top_features(
            df,
            target_col,
            n_top=n_features,
            target_values=target_values,
            embedding_col=embedding_col,
            feature_names=display_names_for(name),
        )
        mat = mat.copy()
        if keys is not None:
            mat.insert(0, "_frame_key", keys.values)
        matrices[name] = mat
        disp[name] = display

    if any(has_keys.values()) and not all(has_keys.values()):
        print(
            "[WARN] Some dataframes lack 'filepath'/'frame_index' — aligning "
            "rows by position instead of frame identity."
        )

    pair_results: dict[tuple[str, str], np.ndarray] = {}
    for name_a, name_b in itertools.combinations(names, 2):
        A = matrices[name_a]
        B = matrices[name_b]
        n_a = len(disp[name_a])
        n_b = len(disp[name_b])

        if has_keys[name_a] and has_keys[name_b]:
            joint = A.merge(B, on="_frame_key", how="inner")
            if joint.shape[0] < 2:
                print(
                    f"[WARN] {name_a} vs {name_b}: no shared frames to correlate."
                )
                continue
            x = joint.iloc[:, 1 : 1 + n_a].to_numpy()
            y = joint.iloc[:, 1 + n_a :].to_numpy()
        elif len(A) == len(B) and len(A) == len(dfs[name_a]):
            x = A.iloc[:, 0:n_a].to_numpy()
            y = B.iloc[:, 0:n_b].to_numpy()
        else:
            print(
                f"[WARN] {name_a} vs {name_b}: rows cannot be aligned, skipping."
            )
            continue

        # --- Pearson correlation between every feature pair ---
        with np.errstate(divide="ignore", invalid="ignore"):
            xz = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
            yz = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
        corr = (yz.T @ xz) / (x.shape[0] - 1)  # (n_b, n_a)
        corr[~np.isfinite(corr)] = 0.0

        # --- plot ---
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr,
            ax=ax,
            cmap=colormap,
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=_thinned_tick_labels(disp[name_a]),
            yticklabels=_thinned_tick_labels(disp[name_b]),
        )
        ax.set_title(
            f"Feature correlation: {name_a} vs {name_b}\n"
            f"top {n_a} vs top {n_b} features by |r| with '{target_col}'"
        )
        ax.set_xlabel(name_a)
        ax.set_ylabel(name_b)
        fig.tight_layout()

        if save_dir is not None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{name_a}_vs_{name_b}_correlation_top{n_features}.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)

        pair_results[(name_a, name_b)] = corr

    return pair_results
