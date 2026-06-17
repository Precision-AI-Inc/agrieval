Module Reference
================

API documentation for the ``pai.ag_emb`` package — metrics, services, schemas, and routes.

pai.ag_emb.metrics
------------------

Top-level exports from the metrics package (similarity, geometry, label-aware, and more).

.. automodule:: pai.ag_emb.metrics
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieval and ranking primitives (nDCG, MAP, MRR, R-Precision, precision, recall).

.. automodule:: pai.ag_emb.metrics.ranking
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pairwise cosine similarity, top-K neighbours, and threshold counts.

.. automodule:: pai.ag_emb.metrics.similarity
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.neighbors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNN diagnostics: hubness, radius, outlier scores, and Gini coefficient.

.. automodule:: pai.ag_emb.metrics.neighbors
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Embedding-space geometry: centroid similarity, PCA explained variance, effective rank, uniformity, and alignment (Wang & Isola 2020).

.. automodule:: pai.ag_emb.metrics.geometry
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.label_aware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Label-aware metrics: intra/inter gap, KNN purity, nDCG, MAP, MRR, R-Precision, metadata-aware retrieval (precision, nDCG, MAP, MRR, R-Precision), per-attribute nDCG, graded relevance, and confusion matrix.

.. automodule:: pai.ag_emb.metrics.label_aware
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.cross_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-model comparison: KNN Jaccard/overlap, similarity correlation, neighbour disagreement.

.. automodule:: pai.ag_emb.metrics.cross_model
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.duplicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Near-duplicate detection at configurable cosine similarity thresholds.

.. automodule:: pai.ag_emb.metrics.duplicates
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.metrics.analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-level convenience wrappers: ``analyze_embedding_space`` and ``compare_embedding_spaces``.

.. automodule:: pai.ag_emb.metrics.analysis
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.services.evaluate
-----------------------------

Service layer: orchestrates metrics into the fixed evaluation JSON schema, supporting both embeddings-only and embeddings+metadata evaluation modes.

.. automodule:: pai.ag_emb.services.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.services.reporting
-------------------------------

Human-readable printing and interactive Plotly visualization of evaluation results.

.. automodule:: pai.ag_emb.services.reporting
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.schemas.evaluate
-----------------------------

Pydantic request and response schemas for the evaluation endpoint, including :class:`MetadataGroup` for explicit-positive similarity groups.

.. automodule:: pai.ag_emb.schemas.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

pai.ag_emb.api.routes.evaluate
--------------------------------

Three named FastAPI endpoints covering the full agricultural retrieval taxonomy:
``POST /v1/embeddings/evaluate/image2image``,
``POST /v1/embeddings/evaluate/plant2image``, and
``POST /v1/embeddings/evaluate/plant2plant``.

.. automodule:: pai.ag_emb.api.routes.evaluate
   :members:
   :undoc-members:
   :show-inheritance:
