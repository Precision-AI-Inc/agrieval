Module Reference
================

API documentation for the ``precisionai.agrieval.emb``, ``precisionai.agrieval.seg``, and
``precisionai.agrieval.dpt`` packages — metrics, services, schemas, and routes.

precisionai.agrieval.emb
------------------------

Top-level package re-exports — the primary entry points for embedding evaluation:
``run_image2image_eval``, ``run_plant2image_eval``, ``run_plant2plant_eval``,
``print_result``, the Plotly visualizations (``plot_knn_confusion``,
``plot_cosine_similarity``, ``plot_tsne``, ``plot_lle``), and :class:`MetadataGroup`.

.. automodule:: precisionai.agrieval.emb
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics
--------------------------------

Top-level exports from the metrics package (similarity, geometry, label-aware, and more).

.. automodule:: precisionai.agrieval.emb.metrics
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieval and ranking primitives (nDCG, MAP, MRR, R-Precision, precision, recall).

.. automodule:: precisionai.agrieval.emb.metrics.ranking
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pairwise cosine similarity, top-K neighbours, and threshold counts.

.. automodule:: precisionai.agrieval.emb.metrics.similarity
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.neighbors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNN diagnostics: hubness, radius, outlier scores, and Gini coefficient.

.. automodule:: precisionai.agrieval.emb.metrics.neighbors
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Embedding-space geometry: centroid similarity, PCA explained variance, effective rank, uniformity, and alignment (Wang & Isola 2020).

.. automodule:: precisionai.agrieval.emb.metrics.geometry
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.label_aware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Label-aware metrics: intra/inter gap, KNN purity, nDCG, MAP, MRR, R-Precision, metadata-aware retrieval (precision, nDCG, MAP, MRR, R-Precision), per-attribute nDCG, graded relevance, and confusion matrix.

.. automodule:: precisionai.agrieval.emb.metrics.label_aware
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.cross_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-model comparison: KNN Jaccard/overlap, similarity correlation, neighbour disagreement.

.. automodule:: precisionai.agrieval.emb.metrics.cross_model
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.duplicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Near-duplicate detection at configurable cosine similarity thresholds.

.. automodule:: precisionai.agrieval.emb.metrics.duplicates
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-level convenience wrappers: ``analyze_embedding_space`` and ``compare_embedding_spaces``.

.. automodule:: precisionai.agrieval.emb.metrics.analysis
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.services.evaluate
------------------------------------------

Service layer: orchestrates metrics into the fixed evaluation JSON schema, supporting both embeddings-only and embeddings+metadata evaluation modes.

.. automodule:: precisionai.agrieval.emb.services.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.services.labels
-----------------------------------------

Label extraction from paths, metadata wiring (``build_image_items``, plant wiring adapters), and the ``image2image.json`` metadata loader.

.. automodule:: precisionai.agrieval.emb.services.labels
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.services.knn_metrics
-----------------------------------------------

KNN metric bundles, per-class breakdowns, global-metric assembly, and HDBSCAN group analysis.

.. automodule:: precisionai.agrieval.emb.services.knn_metrics
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.services.serialization
-------------------------------------------------

JSON serialisation helpers shared by the evaluation services.

.. automodule:: precisionai.agrieval.emb.services.serialization
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.services.reporting
-------------------------------------------

Human-readable printing and interactive Plotly visualization of evaluation results.

.. automodule:: precisionai.agrieval.emb.services.reporting
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.schemas.evaluate
-----------------------------------------

Pydantic request and response schemas for the evaluation endpoint, including :class:`MetadataGroup` for explicit-positive similarity groups.

.. automodule:: precisionai.agrieval.emb.schemas.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.api.routes.evaluate
--------------------------------------------

Three retrieval evaluation endpoints:
``POST /v1/embeddings/evaluate/image2image``,
``POST /v1/embeddings/evaluate/plant2image``, and
``POST /v1/embeddings/evaluate/plant2plant``.

.. automodule:: precisionai.agrieval.emb.api.routes.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.seg
------------------------

Top-level package re-exports — ``run_seg_eval`` and ``load_classes``.

.. automodule:: precisionai.agrieval.seg
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.seg.metrics.segmentation
---------------------------------------------

Pure pixel-level segmentation metrics computed from confusion matrices: per-class IoU,
Dice/F1, accuracy, and the dataset-level mIoU, mAcc, and FWIoU aggregates.

.. automodule:: precisionai.agrieval.seg.metrics.segmentation
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.seg.services.evaluate
------------------------------------------

Service layer: loads colour-coded prediction and ground-truth masks, validates them
against a class definition file, computes per-image and dataset-level KPIs, and writes
the results to JSON.

.. automodule:: precisionai.agrieval.seg.services.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.seg.schemas.evaluate
-----------------------------------------

Pydantic request and response schemas for ``POST /v1/segmentation/evaluate``.

.. automodule:: precisionai.agrieval.seg.schemas.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.seg.api.routes.evaluate
--------------------------------------------

The segmentation evaluation endpoint: ``POST /v1/segmentation/evaluate``.

.. automodule:: precisionai.agrieval.seg.api.routes.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.seg.cli
----------------------------

Command-line interface for semantic segmentation evaluation (``precisionai-agrieval-seg``).

.. automodule:: precisionai.agrieval.seg.cli
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.dpt
------------------------

Top-level package re-exports — ``run_dpt_eval``, ``load_tiles``, ``load_tile_placement``,
:class:`TilePlacement`, ``run_dpt_image_eval``, ``load_images``, and ``print_result``.

.. automodule:: precisionai.agrieval.dpt
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.dpt.metrics.tokens
----------------------------------------

Dense patch token diagnostics: per-patch norm statistics, spatial smoothness, and outlier fraction.

.. automodule:: precisionai.agrieval.dpt.metrics.tokens
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.dpt.services.evaluate
-------------------------------------------

Service layer: loads tiles or whole-image feature maps from batched ``.npz`` archives
(``load_tiles``/``load_images``), flattens them into a patch-token matrix, reuses
``emb.metrics.geometry`` for unsupervised diagnostics, and optionally aligns ground-truth
masks to the patch grid for label-aware separation metrics — ``load_tile_placement`` and
``run_dpt_eval``'s ``tile_placement`` argument enable crop-aware alignment against one
whole-image mask per source image. ``run_dpt_image_eval`` delegates to ``run_dpt_eval``
and relabels the response for the whole-image wiring.

.. automodule:: precisionai.agrieval.dpt.services.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.dpt.services.reporting
--------------------------------------------

Human-readable printing of dense patch token evaluation results (``print_result``),
covering both the tile-keyed and whole-image-keyed result shapes.

.. automodule:: precisionai.agrieval.dpt.services.reporting
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.dpt.schemas.evaluate
-------------------------------------------

Pydantic request and response schemas for ``POST /v1/dense-patch-tokens/evaluate/tiles``
(``DptEvalRequest``/``DptEvalResponse``) and ``POST /v1/dense-patch-tokens/evaluate/image``
(``DptImageEvalRequest``/``DptImageEvalResponse``).

.. automodule:: precisionai.agrieval.dpt.schemas.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.dpt.api.routes.evaluate
-----------------------------------------------

The dense patch token evaluation endpoints: ``POST /v1/dense-patch-tokens/evaluate/tiles`` and
``POST /v1/dense-patch-tokens/evaluate/image``.

.. automodule:: precisionai.agrieval.dpt.api.routes.evaluate
   :members:
   :undoc-members:
   :show-inheritance:
