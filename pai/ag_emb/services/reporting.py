# ======================================================================
#  CONFIDENTIAL — © Precision AI 2025. All Rights Reserved.
#
#  This source code and any accompanying documentation contain
#  confidential and proprietary information of Precision AI.
#
#  Unauthorized reproduction, disclosure, modification, or distribution
#  of this material is strictly prohibited and will be prosecuted to the
#  fullest extent of the law.
# ======================================================================

"""Human-readable display and visualization of evaluation results."""

from __future__ import annotations

from typing import Any

import numpy as np

from pai.ag_emb.services.evaluate import extract_labels

try:
    import plotly.graph_objects as go  # type: ignore[import]

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    from sklearn.manifold import TSNE, LocallyLinearEmbedding  # type: ignore[import]

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def print_result(result: dict) -> None:
    """Pretty-print the output of :func:`~pai.ag_emb.services.evaluate.run_evaluation`.

    Writes a human-readable summary to stdout, including a header line with
    dataset dimensions, global metrics, and per-class breakdowns.

    Parameters
    ----------
    result : dict
        Return value of ``run_evaluation()``.
    """
    print(f"n_items      : {result['n_items']}")
    print(f"embedding_dim: {result['embedding_dim']}")
    print(f"classes      : {result['classes']}")
    print(f"k_values     : {result['k_values']}")
    print()

    _print_global(result["global_metrics"])
    print()
    _print_per_class(result["per_class"])


def _print_global(gm: dict) -> None:
    """Print the global metrics section of an evaluation result.

    Parameters
    ----------
    gm : dict
        The ``global_metrics`` sub-dict from a ``run_evaluation()`` result.
    """
    print("── global_metrics ──────────────────────────────────────────────────────")

    # Pairwise cosine similarity across all embeddings
    ps = gm["pairwise_similarity_stats"]
    print(
        f"  pairwise cosine    : mean={ps['mean']:.4f}  std={ps['std']:.4f}"
        f"  (p05={ps['p05']:.4f}  p50={ps['p50']:.4f}  p95={ps['p95']:.4f})"
    )

    # Cosine similarity of each embedding to the dataset centroid (anisotropy signal)
    cs = gm["centroid_similarity_stats"]
    if cs["mean_cosine_to_centroid"] is not None:
        print(
            f"  centroid cosine    : mean={cs['mean_cosine_to_centroid']:.4f}"
            f"  std={cs['std_cosine_to_centroid']:.4f}"
            f"  norm={cs['centroid_norm']:.4f}"
        )

    # Intra vs inter-class separation
    gap = gm["intra_inter_similarity_gap"]
    if gap["gap"] is not None:
        print(
            f"  intra/inter gap    : {gap['gap']:.4f}"
            f"  (intra={gap['mean_intra_class_similarity']:.4f}"
            f"  inter={gap['mean_inter_class_similarity']:.4f})"
        )

    print()

    # KNN label purity: fraction of k neighbors sharing the same class label
    for k, stats in gm["knn_label_purity"].items():
        print(f"  KNN purity@{k:<4}   : mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    # nDCG: ranking quality — relevant (same-class) neighbors ranked first scores higher
    for k, stats in gm["knn_label_ndcg"].items():
        print(f"  nDCG@{k:<10}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    # MAP: mean average precision — rewards finding all relevant neighbors early
    for k, stats in gm["knn_map"].items():
        print(f"  MAP@{k:<11}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    print()

    er = gm["effective_rank"]
    if er["effective_rank"] is not None:
        print(
            f"  effective_rank     : {er['effective_rank']:.2f}"
            f"  (ratio={er['effective_rank_ratio']:.4f},"
            f"  dim={er['embedding_dim']})"
        )


def _print_per_class(per_class: dict) -> None:
    """Print the per-class metrics section of an evaluation result.

    Parameters
    ----------
    per_class : dict
        The ``per_class`` sub-dict from a ``run_evaluation()`` result, keyed
        by crop class name.
    """
    print("── per_class ───────────────────────────────────────────────────────────")
    for cls, m in per_class.items():
        print(f"\n  [{cls}]  n={m['n_items']}")

        if "pairwise_similarity_stats" in m:
            ps = m["pairwise_similarity_stats"]
            print(f"    pairwise cosine  : mean={ps['mean']:.4f}  std={ps['std']:.4f}")

        if "centroid_similarity_stats" in m:
            cs = m["centroid_similarity_stats"]
            if cs["mean_cosine_to_centroid"] is not None:
                print(
                    f"    centroid cosine  : mean={cs['mean_cosine_to_centroid']:.4f}"
                    f"  norm={cs['centroid_norm']:.4f}"
                )

        if "effective_rank" in m:
            er = m["effective_rank"]
            if er.get("effective_rank") is not None:
                print(
                    f"    effective_rank   : {er['effective_rank']:.2f}" f"  (ratio={er['effective_rank_ratio']:.4f})"
                )

        for k, stats in m.get("knn_label_purity", {}).items():
            print(
                f"    KNN purity@{k:<4}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                f"  (p05={stats['p05']:.4f}  p95={stats['p95']:.4f})"
            )

        for k, stats in m.get("knn_label_ndcg", {}).items():
            print(
                f"    nDCG@{k:<9}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                f"  (p05={stats['p05']:.4f}  p95={stats['p95']:.4f})"
            )

        for k, stats in m.get("knn_map", {}).items():
            print(
                f"    MAP@{k:<10}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                f"  (p05={stats['p05']:.4f}  p95={stats['p95']:.4f})"
            )


# ---------------------------------------------------------------------------
# Visualizations (Plotly — interactive HTML by default)
# ---------------------------------------------------------------------------


def _write_plotly(fig: Any, output_path: str) -> None:
    """Write a Plotly figure to an HTML or static-image file.

    HTML output uses a CDN-hosted Plotly bundle.  Static image formats
    (``.png``, ``.pdf``, etc.) require the ``kaleido`` package; if it is
    absent the figure is saved as ``.html`` instead and a notice is printed.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to write.
    output_path : str
        Destination file path.  The extension determines the format.
    """
    if output_path.lower().endswith(".html"):
        fig.write_html(output_path, include_plotlyjs="cdn")
    else:
        try:
            fig.write_image(output_path)
        except Exception:
            html_path = output_path.rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"  (kaleido not available — saved as {html_path} instead)")
            output_path = html_path
    print(f"Saved: {output_path}")


def plot_knn_confusion(
    result: dict,
    k: int | None = None,
    output_path: str = "class_confusion_matrix.html",
) -> None:
    """Save an interactive KNN confusion matrix heatmap (Plotly).

    Rows = true class, columns = neighbor class, values = fraction of k-NN
    neighbors belonging to each class.  The diagonal equals mean KNN purity.

    Can be generated directly from a saved JSON result (no raw vectors needed).

    Parameters
    ----------
    result : dict
        Output of ``run_evaluation()`` or a JSON-deserialized equivalent.
    k : int | None
        K cutoff to visualize.  Defaults to the largest k in the result.
    output_path : str
        Destination file path.  ``.html`` (default) produces an interactive page;
        ``.png``/``.pdf`` requires ``kaleido`` (``pip install kaleido``).
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required: pip install plotly") from None

    if k is None:
        k = result["k_values"][-1]

    confusion = result.get("knn_confusion", {}).get(str(k))
    if confusion is None:
        raise ValueError(f"knn_confusion not found for k={k}. Re-run run_evaluation() to regenerate.")

    classes = result["classes"]
    n_cls = len(classes)
    matrix = np.array([[confusion[c1][c2] for c2 in classes] for c1 in classes])
    text = [[f"{matrix[i, j]:.3f}" for j in range(n_cls)] for i in range(n_cls)]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.tolist(),
            x=classes,
            y=classes,
            colorscale="RdYlGn",
            zmin=0.0,
            zmax=1.0,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            hovertemplate=(
                "True class: <b>%{y}</b><br>" "Neighbor class: <b>%{x}</b><br>" "Fraction: %{z:.4f}<extra></extra>"
            ),
            colorbar=dict(title=f"KNN@{k}<br>neighbor<br>fraction", thickness=18),
        )
    )

    cell_px = max(90, min(200, 600 // n_cls))
    fig.update_layout(
        title=dict(text=f"KNN Confusion Matrix  (k={k})", font=dict(size=17)),
        xaxis=dict(title="Neighbor class", side="bottom", tickfont=dict(size=12)),
        yaxis=dict(title="True class", autorange="reversed", tickfont=dict(size=12)),
        width=max(480, n_cls * cell_px + 160),
        height=max(400, n_cls * cell_px + 130),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=120, r=60, t=70, b=100),
    )

    _write_plotly(fig, output_path)


def plot_cosine_similarity(
    image_embeddings: dict,
    result: dict,
    output_path: str = "cosine_similarity.html",
) -> None:
    """Save an interactive pairwise cosine similarity heatmap (Plotly).

    Embeddings are sorted by class label so within-class blocks appear on the
    diagonal.  Hover shows the two image paths and their cosine similarity.

    Parameters
    ----------
    image_embeddings : dict
        The same ``{path: vector}`` dict passed to ``run_evaluation()``.
    result : dict
        Output of ``run_evaluation()``.
    output_path : str
        Destination file path.  ``.html`` produces an interactive page.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required: pip install plotly") from None

    paths = list(image_embeddings.keys())
    vectors = np.array(list(image_embeddings.values()), dtype=np.float32)
    n = len(vectors)

    item_labels = result.get("item_labels")
    if item_labels is None:
        item_labels = extract_labels(paths)

    label_arr = np.array(item_labels)
    classes = result["classes"]

    # Sort by class so within-class blocks sit on the diagonal
    sort_order = np.argsort(label_arr, kind="stable")
    sorted_labels = label_arr[sort_order]
    sorted_paths = [paths[i] for i in sort_order]
    sorted_vectors = vectors[sort_order]

    # L2-normalised → cosine sim = dot product
    sim = (sorted_vectors @ sorted_vectors.T).astype(np.float64)
    np.clip(sim, -1.0, 1.0, out=sim)

    tick_labels = [p.split("/")[-1] for p in sorted_paths]

    fig = go.Figure(
        data=go.Heatmap(
            z=sim.tolist(),
            x=tick_labels,
            y=tick_labels,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            hovertemplate=("Row: %{y}<br>Col: %{x}<br>Cosine similarity: %{z:.4f}<extra></extra>"),
            colorbar=dict(title="Cosine<br>similarity", thickness=18),
        )
    )

    # Class boundary lines
    shapes = []
    prev = 0
    for cls in classes:
        count = int(np.sum(sorted_labels == cls))
        if prev > 0:
            boundary = prev - 0.5
            for is_vertical in (True, False):
                shapes.append(
                    dict(
                        type="line",
                        x0=boundary if is_vertical else -0.5,
                        x1=boundary if is_vertical else n - 0.5,
                        y0=boundary if not is_vertical else -0.5,
                        y1=boundary if not is_vertical else n - 0.5,
                        line=dict(color="black", width=2),
                    )
                )
        prev += count

    cell_px = max(18, min(40, 800 // n))
    fig.update_layout(
        title=dict(
            text=f"Pairwise Cosine Similarity  (n={n}, sorted by class)",
            font=dict(size=17),
        ),
        xaxis=dict(tickfont=dict(size=max(7, 11 - n // 10)), tickangle=45),
        yaxis=dict(tickfont=dict(size=max(7, 11 - n // 10)), autorange="reversed"),
        shapes=shapes,
        width=max(500, n * cell_px + 180),
        height=max(450, n * cell_px + 160),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=140, r=60, t=70, b=140),
    )

    _write_plotly(fig, output_path)


def _build_scatter3d(
    fig: Any,
    coords: Any,
    classes: list,
    label_arr: Any,
    paths: list,
    axis_prefix: str,
) -> None:
    """Add one ``Scatter3d`` trace per class to an existing Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to which traces are appended in-place.
    coords : array-like
        Shape ``[N, 3]`` low-dimensional coordinates (e.g., t-SNE or LLE).
    classes : list
        Ordered list of unique class names.
    label_arr : array-like
        Shape ``[N]`` class label per item (same order as ``paths``).
    paths : list[str]
        Image paths used as hover text.
    axis_prefix : str
        Short string prepended to axis labels in hover tooltips (e.g.,
        ``"t-SNE"`` or ``"LLE"``).
    """
    n = len(paths)
    for cls in classes:
        mask = label_arr == cls
        cls_paths = [paths[i] for i in range(n) if mask[i]]
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0].tolist(),
                y=coords[mask, 1].tolist(),
                z=coords[mask, 2].tolist(),
                mode="markers",
                name=cls,
                text=cls_paths,
                hovertemplate="%{text}<extra>" + cls + "</extra>",
                marker=dict(size=7, opacity=0.85, line=dict(width=1, color="white")),
            )
        )


def plot_lle(
    image_embeddings: dict,
    result: dict,
    output_path: str = "lle.html",
    n_neighbors: int | None = None,
) -> None:
    """Save an interactive 3D LLE (Locally Linear Embedding) scatter (Plotly).

    LLE preserves local neighbourhood structure rather than global distances,
    complementing t-SNE.  Requires ``scikit-learn``.

    Parameters
    ----------
    image_embeddings : dict
        The same ``{path: vector}`` dict passed to ``run_evaluation()``.
    result : dict
        Output of ``run_evaluation()``.
    output_path : str
        Destination file path.  ``.html`` produces a rotatable 3D page.
    n_neighbors : int | None
        LLE neighbourhood size.  Defaults to ``max(5, n // 3)`` clamped to
        ``n - 1``.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required: pip install plotly") from None
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for LLE: pip install scikit-learn") from None

    paths = list(image_embeddings.keys())
    vectors = np.array(list(image_embeddings.values()), dtype=np.float32)
    n = len(vectors)

    item_labels = result.get("item_labels")
    if item_labels is None:
        item_labels = extract_labels(paths)

    classes = result["classes"]
    label_arr = np.array(item_labels)

    k = min(n - 1, n_neighbors if n_neighbors is not None else max(5, n // 3))
    coords = LocallyLinearEmbedding(
        n_components=3,
        n_neighbors=k,
        random_state=42,
    ).fit_transform(vectors.astype(np.float64))

    fig = go.Figure()
    _build_scatter3d(fig, coords, classes, label_arr, paths, axis_prefix="LLE")

    fig.update_layout(
        title=dict(text=f"LLE 3D  (n={n},  n_neighbors={k})", font=dict(size=17)),
        scene=dict(
            xaxis_title="LLE 1",
            yaxis_title="LLE 2",
            zaxis_title="LLE 3",
            bgcolor="white",
        ),
        legend=dict(title="Class", font=dict(size=12)),
        width=950,
        height=750,
        paper_bgcolor="white",
    )

    _write_plotly(fig, output_path)


def plot_tsne(
    image_embeddings: dict,
    result: dict,
    output_path: str = "tsne.html",
    perplexity: int = 30,
    dimensions: int = 3,
) -> None:
    """Save an interactive t-SNE scatter plot of embeddings colored by class (Plotly).

    Requires ``scikit-learn`` for the t-SNE computation.  Cannot be generated
    from a saved JSON result alone — the raw embedding vectors are needed.

    Parameters
    ----------
    image_embeddings : dict
        The same ``{path: vector}`` dict passed to ``run_evaluation()``.
    result : dict
        Output of ``run_evaluation()``.
    output_path : str
        Destination file path.  ``.html`` (default) produces an interactive page;
        ``.png``/``.pdf`` requires ``kaleido``.
    perplexity : int
        t-SNE perplexity.  Auto-clamped to ``max(5, n // 3)`` for small datasets.
    dimensions : int
        2 or 3.  3D produces a rotatable scatter; 2D is simpler for dense datasets.
    """
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")

    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required: pip install plotly") from None
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for t-SNE: pip install scikit-learn") from None

    paths = list(image_embeddings.keys())
    vectors = np.array(list(image_embeddings.values()), dtype=np.float32)
    n = len(vectors)

    item_labels = result.get("item_labels")
    if item_labels is None:
        item_labels = extract_labels(paths)

    classes = result["classes"]
    label_arr = np.array(item_labels)

    effective_perplexity = min(perplexity, max(5, n // 3))
    coords = TSNE(
        n_components=dimensions,
        perplexity=effective_perplexity,
        random_state=42,
    ).fit_transform(vectors.astype(np.float64))

    title_text = f"t-SNE {dimensions}D  (n={n},  perplexity={effective_perplexity})"
    fig = go.Figure()

    if dimensions == 3:
        _build_scatter3d(fig, coords, classes, label_arr, paths, axis_prefix="t-SNE")
    else:
        for cls in classes:
            mask = label_arr == cls
            cls_paths = [paths[i] for i in range(n) if mask[i]]
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0].tolist(),
                    y=coords[mask, 1].tolist(),
                    mode="markers",
                    name=cls,
                    text=cls_paths,
                    hovertemplate="%{text}<extra>" + cls + "</extra>",
                    marker=dict(size=10, opacity=0.85, line=dict(width=1, color="white")),
                )
            )

    if dimensions == 3:
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=17)),
            scene=dict(
                xaxis_title="t-SNE 1",
                yaxis_title="t-SNE 2",
                zaxis_title="t-SNE 3",
                bgcolor="white",
            ),
            legend=dict(title="Class", font=dict(size=12)),
            width=950,
            height=750,
            paper_bgcolor="white",
        )
    else:
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=17)),
            xaxis=dict(title="t-SNE 1", showgrid=True, gridcolor="#e8e8e8", zeroline=False),
            yaxis=dict(title="t-SNE 2", showgrid=True, gridcolor="#e8e8e8", zeroline=False),
            legend=dict(title="Class", font=dict(size=12)),
            width=950,
            height=680,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

    _write_plotly(fig, output_path)
