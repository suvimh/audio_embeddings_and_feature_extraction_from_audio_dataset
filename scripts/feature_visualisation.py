# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap


def lda_visualisation(x, y, data, palette="viridis", data_type="VGGish Features"):
    x_flat = x.reshape(x.shape[0], -1)
    lda = LDA(n_components=2)
    embedding = lda.fit_transform(x_flat, y)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], hue=data["Class"], palette=palette
    )
    plt.title(f"LDA Projection of {data_type}")
    plt.show()


def pca_visualisation(x, data, palette="viridis", data_type="VGGish Features"):
    x_flat = x.reshape(x.shape[0], -1)
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(x_flat)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], hue=data["Class"], palette=palette
    )
    plt.title(f"PCA Projection of {data_type}")
    plt.show()


def tsne_visualisation(x, y, encoder, palette="viridis", data_type="VGGish Features"):
    x_flat = x.reshape(x.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    x_2d = tsne.fit_transform(x_flat)

    # Create a color palette of the right length
    class_names = encoder.classes_
    n_classes = len(class_names)
    colours = sns.color_palette(palette, n_colors=n_classes)

    plt.figure(figsize=(10, 8))

    # Plot each class individually to ensure colors match legend
    for i, class_name in enumerate(class_names):
        idx = np.where(y == i)[0]
        plt.scatter(
            x_2d[idx, 0], x_2d[idx, 1], color=colours[i], alpha=0.7, label=class_name
        )

    plt.legend(title="Classes", loc="best")
    plt.title(f"t-SNE Visualization of {data_type}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def umap_visualisation(x, y, encoder, palette="viridis", data_type="Whisper Features"):
    import umap

    x_flat = x.reshape(x.shape[0], -1)
    reducer = umap.UMAP(n_components=2, random_state=42)
    x_2d = reducer.fit_transform(x_flat)

    # Create a color palette of the right length
    class_names = encoder.classes_
    n_classes = len(class_names)
    colors = sns.color_palette(palette, n_colors=n_classes)

    plt.figure(figsize=(10, 8))

    # Plot each class separately to ensure color matches legend
    for i, class_name in enumerate(class_names):
        idx = np.where(y == i)[0]
        plt.scatter(
            x_2d[idx, 0], x_2d[idx, 1], color=colors[i], alpha=0.7, label=class_name
        )

    plt.legend(title="Classes", loc="best")
    plt.title(f"UMAP Visualization of {data_type}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def pca_visualisation_3d(x, data, palette="viridis", data_type="VGGish Features"):
    x_flat = x.reshape(x.shape[0], -1)
    pca = PCA(n_components=3)
    embedding = pca.fit_transform(x_flat)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create a color palette
    class_names = data["Class"].unique()
    colours = sns.color_palette(palette, n_colors=len(class_names))

    # Plot each class separately
    for i, class_name in enumerate(class_names):
        idx = np.where(data["Class"] == class_name)[0]
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            embedding[idx, 2],
            color=colours[i],
            alpha=0.7,
            label=class_name,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA Projection of {data_type}")
    ax.legend(title="Classes", loc="best")
    plt.show()


def lda_visualisation_3d(x, y, data, palette="viridis", data_type="VGGish Features"):
    x_flat = x.reshape(x.shape[0], -1)
    n_components = min(3, len(np.unique(y)) - 1)
    lda = LDA(n_components=n_components)
    embedding = lda.fit_transform(x_flat, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create a color palette
    class_names = data["Class"].unique()
    colours = sns.color_palette(palette, n_colors=len(class_names))

    # Plot each class separately
    for i, class_name in enumerate(class_names):
        idx = np.where(data["Class"] == class_name)[0]
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            embedding[idx, 2] if n_components > 2 else np.zeros_like(embedding[idx, 0]),
            color=colours[i],
            alpha=0.7,
            label=class_name,
        )

    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    if n_components > 2:
        ax.set_zlabel("LD3")
    ax.set_title(f"3D LDA Projection of {data_type}")
    ax.legend(title="Classes", loc="best")
    plt.show()


def tsne_visualisation_3d(
    x, y, encoder, palette="viridis", data_type="Whisper Features"
):
    x_flat = x.reshape(x.shape[0], -1)
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    x_3d = tsne.fit_transform(x_flat)

    # Create a color palette of the right length
    class_names = encoder.classes_
    n_classes = len(class_names)
    colours = sns.color_palette(palette, n_colors=n_classes)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each class separately
    for i, class_name in enumerate(class_names):
        idx = np.where(y == i)[0]
        ax.scatter(
            x_3d[idx, 0],
            x_3d[idx, 1],
            x_3d[idx, 2],
            color=colours[i],
            alpha=0.7,
            label=class_name,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.set_title(f"3D t-SNE Visualization of {data_type}")
    ax.legend(title="Classes", loc="best")
    plt.show()


def umap_visualisation_3d(
    x, y, encoder, palette="viridis", data_type="Whisper Features"
):
    x_flat = x.reshape(x.shape[0], -1)
    reducer = umap.UMAP(n_components=3, random_state=42)
    x_3d = reducer.fit_transform(x_flat)

    # Create a color palette of the right length
    class_names = encoder.classes_
    n_classes = len(class_names)
    colours = sns.color_palette(palette, n_colors=n_classes)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each class separately
    for i, class_name in enumerate(class_names):
        idx = np.where(y == i)[0]
        ax.scatter(
            x_3d[idx, 0],
            x_3d[idx, 1],
            x_3d[idx, 2],
            color=colours[i],
            alpha=0.7,
            label=class_name,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title(f"3D UMAP Visualization of {data_type}")
    ax.legend(title="Classes", loc="best")
    plt.show()
