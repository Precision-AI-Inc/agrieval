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


def print_result(result: dict) -> None:
    """Pretty-print the output of :func:`~pai.ag_emb.services.evaluate.run_evaluation`."""
    print(f"n_items      : {result['n_items']}")
    print(f"embedding_dim: {result['embedding_dim']}")
    print(f"classes      : {result['classes']}")
    print(f"k_values     : {result['k_values']}")
    print()

    _print_global(result["global_metrics"])
    print()
    _print_per_class(result["per_class"])


def _print_global(gm: dict) -> None:
    print("── global_metrics ──────────────────────────────────────────────────────")

    # Pairwise cosine similarity across all embeddings
    ps = gm["pairwise_similarity_stats"]
    print(f"  pairwise cosine    : mean={ps['mean']:.4f}  std={ps['std']:.4f}"
          f"  (p05={ps['p05']:.4f}  p50={ps['p50']:.4f}  p95={ps['p95']:.4f})")

    # Cosine similarity of each embedding to the dataset centroid (anisotropy signal)
    cs = gm["centroid_similarity_stats"]
    if cs["mean_cosine_to_centroid"] is not None:
        print(f"  centroid cosine    : mean={cs['mean_cosine_to_centroid']:.4f}"
              f"  std={cs['std_cosine_to_centroid']:.4f}"
              f"  norm={cs['centroid_norm']:.4f}")

    # Intra vs inter-class separation
    gap = gm["intra_inter_similarity_gap"]
    if gap["gap"] is not None:
        print(f"  intra/inter gap    : {gap['gap']:.4f}"
              f"  (intra={gap['mean_intra_class_similarity']:.4f}"
              f"  inter={gap['mean_inter_class_similarity']:.4f})")

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
        print(f"  effective_rank     : {er['effective_rank']:.2f}"
              f"  (ratio={er['effective_rank_ratio']:.4f},"
              f"  dim={er['embedding_dim']})")


def _print_per_class(per_class: dict) -> None:
    print("── per_class ───────────────────────────────────────────────────────────")
    for cls, m in per_class.items():
        print(f"\n  [{cls}]  n={m['n_items']}")

        if "pairwise_similarity_stats" in m:
            ps = m["pairwise_similarity_stats"]
            print(f"    pairwise cosine  : mean={ps['mean']:.4f}  std={ps['std']:.4f}")

        if "centroid_similarity_stats" in m:
            cs = m["centroid_similarity_stats"]
            if cs["mean_cosine_to_centroid"] is not None:
                print(f"    centroid cosine  : mean={cs['mean_cosine_to_centroid']:.4f}"
                      f"  norm={cs['centroid_norm']:.4f}")

        if "effective_rank" in m:
            er = m["effective_rank"]
            if er.get("effective_rank") is not None:
                print(f"    effective_rank   : {er['effective_rank']:.2f}"
                      f"  (ratio={er['effective_rank_ratio']:.4f})")

        for k, stats in m.get("knn_label_purity", {}).items():
            print(f"    KNN purity@{k:<4}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                  f"  (p05={stats['p05']:.4f}  p95={stats['p95']:.4f})")

        for k, stats in m.get("knn_label_ndcg", {}).items():
            print(f"    nDCG@{k:<9}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                  f"  (p05={stats['p05']:.4f}  p95={stats['p95']:.4f})")

        for k, stats in m.get("knn_map", {}).items():
            print(f"    MAP@{k:<10}  : mean={stats['mean']:.4f}  std={stats['std']:.4f}"
                  f"  (p05={stats['p05']:.4f}  p95={stats['p95']:.4f})")


# ---------------------------------------------------------------------------
# Visualizations (Plotly — interactive HTML by default)
# ---------------------------------------------------------------------------

def _write_plotly(fig: object, output_path: str) -> None:
    """Write a Plotly figure to HTML or static image (requires kaleido for non-HTML)."""
    if output_path.lower().endswith(".html"):
        fig.write_html(output_path, include_plotlyjs="cdn")  # type: ignore[attr-defined]
    else:
        try:
            fig.write_image(output_path)  # type: ignore[attr-defined]
        except Exception:
            html_path = output_path.rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path, include_plotlyjs="cdn")  # type: ignore[attr-defined]
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
    try:
        import numpy as np
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    if k is None:
        k = result["k_values"][-1]

    confusion = result.get("knn_confusion", {}).get(str(k))
    if confusion is None:
        raise ValueError(
            f"knn_confusion not found for k={k}. Re-run run_evaluation() to regenerate."
        )

    classes = result["classes"]
    n_cls = len(classes)
    matrix = np.array([[confusion[c1][c2] for c2 in classes] for c1 in classes])
    text = [[f"{matrix[i, j]:.3f}" for j in range(n_cls)] for i in range(n_cls)]

    fig = go.Figure(data=go.Heatmap(
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
            "True class: <b>%{y}</b><br>"
            "Neighbor class: <b>%{x}</b><br>"
            "Fraction: %{z:.4f}<extra></extra>"
        ),
        colorbar=dict(title=f"KNN@{k}<br>neighbor<br>fraction", thickness=18),
    ))

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

    try:
        import numpy as np
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn is required for t-SNE: pip install scikit-learn")

    paths = list(image_embeddings.keys())
    vectors = np.array(list(image_embeddings.values()), dtype=np.float32)
    n = len(vectors)

    item_labels = result.get("item_labels")
    if item_labels is None:
        from pai.ag_emb.services.evaluate import extract_labels
        item_labels = extract_labels(paths)

    classes = result["classes"]
    label_arr = np.array(item_labels)

    effective_perplexity = min(perplexity, max(5, n // 3))
    coords = TSNE(
        n_components=dimensions, perplexity=effective_perplexity, random_state=42,
    ).fit_transform(vectors.astype(np.float64))

    title_text = (
        f"t-SNE {dimensions}D  (n={n},  perplexity={effective_perplexity})"
    )
    fig = go.Figure()

    for cls in classes:
        mask = label_arr == cls
        cls_paths = [paths[i] for i in range(n) if mask[i]]
        marker = dict(size=7, opacity=0.85, line=dict(width=1, color="white"))
        hover = "%{text}<extra>" + cls + "</extra>"

        if dimensions == 3:
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0].tolist(),
                y=coords[mask, 1].tolist(),
                z=coords[mask, 2].tolist(),
                mode="markers",
                name=cls,
                text=cls_paths,
                hovertemplate=hover,
                marker=marker,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=coords[mask, 0].tolist(),
                y=coords[mask, 1].tolist(),
                mode="markers",
                name=cls,
                text=cls_paths,
                hovertemplate=hover,
                marker=dict(size=10, opacity=0.85, line=dict(width=1, color="white")),
            ))

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
