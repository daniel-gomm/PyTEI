PyTEI package
=============

The main ``TEIClient`` class
*****************************

The `TEIClient` is used to interface with a Text Embeddings Inference instance and is thus the main user-facing class.

.. autoclass:: pytei.TEIClient
   :members:
   :undoc-members:
   :show-inheritance:


Submodules
----------

The pytei.store module
**********************

Use `EmbeddingStore`s for caching embeddings. The `InMemoryEmbeddingStore` serves as default in-memory cache.
Use the `DuckDBEmbeddingStore` for persistent caching. Custom `EmbeddingStore`s can be implemented by extending the
abstract `EmbeddingStore` base class.

.. automodule:: pytei.store
   :members:
   :undoc-members:
   :show-inheritance:


The pytei.model module
**********************

The model defines the structure of the data returned by the `TEIClient`.

.. automodule:: pytei.model
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: pytei
   :members:
   :undoc-members:
   :show-inheritance:
