import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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

    # Plot
    if mode == "2d" or n_components < 3:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="LD1", y="LD2", hue="Plot_Hue_Label", data=data_plot, palette=palette
        )
        plt.title(f"LDA Projection of {data_type} (2D)")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.show()
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        hue_class_names = data_plot["Plot_Hue_Label"].unique()
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
