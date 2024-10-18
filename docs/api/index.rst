=============
API Reference
=============

The API reference gives an overview of ``TopoNetX``, which consists of several modules:

- ``classes`` implements the topological domains, e.g., :py:class:`SimplicialComplex <toponetx.SimplicialComplex>`, :py:class:`CellComplex <toponetx.CellComplex>`, :py:class:`CombinatorialComplex <toponetx.CombinatorialComplex>`.
- ``datasets`` implements utilities to load small datasets on topological domains.
- ``transform`` implements functions to transform the topological domain that supports a dataset, effectively "lifting" the dataset onto another domain.
- ``algorithms`` implements signal processing techniques on topological domains, such as the eigendecomposition of a laplacian.
- ``generators`` implements functions to generate random topological domains.
- ``readwrite`` implements functions to read and write topological domains from and to disk.


.. toctree::
   :maxdepth: 2
   :caption: Packages & Modules

   classes
   datasets
   transform
   algorithms
   generators
   readwrite
