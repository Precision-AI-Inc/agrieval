Module Reference
================

API documentation for the ``precisionai.agrieval.emb`` package — metrics, services, schemas, and routes.

precisionai.agrieval.emb.metrics
------------------

Top-level exports from the metrics package (similarity, geometry, label-aware, and more).

.. automodule:: precisionai.agrieval.emb.metrics
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieval and ranking primitives (nDCG, MAP, MRR, R-Precision, precision, recall).

.. automodule:: precisionai.agrieval.emb.metrics.ranking
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pairwise cosine similarity, top-K neighbours, and threshold counts.

.. automodule:: precisionai.agrieval.emb.metrics.similarity
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.neighbors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNN diagnostics: hubness, radius, outlier scores, and Gini coefficient.

.. automodule:: precisionai.agrieval.emb.metrics.neighbors
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Embedding-space geometry: centroid similarity, PCA explained variance, effective rank, uniformity, and alignment (Wang & Isola 2020).

.. automodule:: precisionai.agrieval.emb.metrics.geometry
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.label_aware
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Label-aware metrics: intra/inter gap, KNN purity, nDCG, MAP, MRR, R-Precision, metadata-aware retrieval (precision, nDCG, MAP, MRR, R-Precision), per-attribute nDCG, graded relevance, and confusion matrix.

.. automodule:: precisionai.agrieval.emb.metrics.label_aware
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.cross_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-model comparison: KNN Jaccard/overlap, similarity correlation, neighbour disagreement.

.. automodule:: precisionai.agrieval.emb.metrics.cross_model
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.duplicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Near-duplicate detection at configurable cosine similarity thresholds.

.. automodule:: precisionai.agrieval.emb.metrics.duplicates
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.metrics.analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-level convenience wrappers: ``analyze_embedding_space`` and ``compare_embedding_spaces``.

.. automodule:: precisionai.agrieval.emb.metrics.analysis
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.services.evaluate
-----------------------------

Service layer: orchestrates metrics into the fixed evaluation JSON schema, supporting both embeddings-only and embeddings+metadata evaluation modes.

.. automodule:: precisionai.agrieval.emb.services.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.services.reporting
-------------------------------

Human-readable printing and interactive Plotly visualization of evaluation results.

.. automodule:: precisionai.agrieval.emb.services.reporting
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.schemas.evaluate
-----------------------------

Pydantic request and response schemas for the evaluation endpoint, including :class:`MetadataGroup` for explicit-positive similarity groups.

.. automodule:: precisionai.agrieval.emb.schemas.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agrieval.emb.api.routes.evaluate
--------------------------------

Three named FastAPI endpoints covering the full agricultural retrieval taxonomy:
``POST /v1/embeddings/evaluate/image2image``,
``POST /v1/embeddings/evaluate/plant2image``, and
``POST /v1/embeddings/evaluate/plant2plant``.

.. automodule:: precisionai.agrieval.emb.api.routes.evaluate
   :members:
   :undoc-members:
   :show-inheritance:
