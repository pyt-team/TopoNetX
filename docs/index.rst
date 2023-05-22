🌐 TopoNetX (TNX) 🍩
===================
Computing on Topological Domains
================================

.. figure:: https://user-images.githubusercontent.com/8267869/234068354-af9480f1-1d18-4914-92f1-916d9093e44d.png
   :alt: natural shapes
   :class: with-shadow
   :width: 1000px

Many natural systems as diverse as social networks and proteins are characterized by relational structure. This is the structure of interactions between
components in the system, such as social interactions between individuals or electrostatic interactions between atoms.

`TopoNetX` provides a unifying interface to compute with such relational data.

🎯 Scope and functionality
--------------------------

`TopoNetX` (TNX) is a package for computing with topological domains and studying their properties.

With its dynamic construction capabilities and support for arbitrary
attributes and data, `TopoNetX` allows users to easily explore the topological structure
of their data and gain insights into its underlying geometric and algebraic properties.

Available functionality ranges
from computing boundary operators and Hodge Laplacians on simplicial/cell/combinatorial complexes
to performing higher-order adjacency calculations.

TNX is similar to `NetworkX`, a popular graph package, and extends its capabilities to support a
wider range of mathematical structures, including cell complexes, simplicial complexes and
combinatorial complexes.

The TNX library provides classes and methods for modeling the entities and relations
found in higher-order networks such as simplicial, cellular, CW and combinatorial complexes.
This package serves as a repository of the methods and algorithms we find most useful
as we explore the knowledge that can be encoded via higher-order networks.

TNX supports the construction of topological structures including the `CellComplex`, `SimplicialComplex` and `CombinatorialComplex` classes.

These classes provide methods for computing boundary operators, Hodge Laplacians
and higher-order adjacency operators on cell, simplicial and combinatorial complexes,
respectively. The classes are used in many areas of mathematics and computer science,
such as algebraic topology, geometry, and data analysis.

TNX was developed by the pyt-team.

🛠️ Main features
----------------

1. Dynamic construction of cell, simplicial and combinatorial complexes, allowing users to add or remove objects from these structures after their initial creation.

2. Compatibility with the `NetworkX` and `gudhi` packages, enabling users to
leverage the powerful algorithms and data structures provided by these packages.

3. Support for attaching arbitrary attributes and data to cells, simplices and other entities in a complex, allowing users to store and manipulate a versatile range of information about these objects.

4. Computation of boundary operators, Hodge Laplacians and higher-order adjacency
operators on a complex, enabling users to study the topological properties of the space.

5. Robust error handling and validation of input data, ensuring that the package is
reliable and easy to use.

6. Package dependencies are kept to a minimum,
to facilitate easy installation and
to reduce future installation issues arising from such dependencies.

Want to try it yourself? Check out our tutorials to get started.

.. toctree::
   :maxdepth: 1
   :hidden:

   api/index
   tutorials/index
   contributing/index
