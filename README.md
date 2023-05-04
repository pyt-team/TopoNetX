[![Test](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/pyt-team/TopoNetX/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pyt-team/TopoNetX)

# 🌐 TopoNetX (TNX) 🍩
# Computing on Topological Domains

![toponetx](https://user-images.githubusercontent.com/8267869/234068354-af9480f1-1d18-4914-92f1-916d9093e44d.png)

Many natural systems as diverse as social networks and proteins are characterized by relational structure. This is the structure of interactions between
components in the system, such as social interactions between individuals or electrostatic interactions between atoms.

`TopoNetX` provides a unifying interface to compute with such relational data.

## 🎯 Scope and functionality

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

TNX supports the construction of many topological structures including the `CellComplex`, `SimplicialComplex` and `CombinatorialComplex` classes.
These classes provide methods for computing boundary operators, Hodge Laplacians
and higher-order adjacency operators on cell, simplicial and combinatorial complexes,
respectively. The classes are used in many areas of mathematics and computer science,
such as algebraic topology, geometry, and data analysis.

TNX was developed by the pyt-team.

## 🛠️ Main features

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

# 🤖 Installing TopoNetX

1. Clone a copy of `TopoNetX` from source:
```bash
git clone https://github.com/pyt-team/TopoNetX
cd TopoNetX
```
2. If you have already cloned `TopoNetX` from source, update it:
```bash
git pull
```
3. Install `TopoNetX` in editable mode:
```bash
pip install -e ".[all]"
```
4. Install pre-commit hooks:
```bash
pre-commit install
```

# 🦾 Getting Started

## Example 1: creating a simplicial complex

```python
import toponetx as tnx

# Instantiate a SimplicialComplex object with a few simplices

sc = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

# Compute the incidence matrix between 1-skeleton and 0-skeleton

B1 = sc.incidence_matrix(1)

# Compute the incidence matrix between 2-skeleton and 1-skeleton

B2 = sc.incidence_matrix(2)
```

## Example 2: creating a cell complex

```python
import toponetx as tnx

# Instantiate a CellComplex object with a few cells

cx = tnx.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

# Add an edge (cell of rank 1) after initialization

cx.add_edge(0, 1)

# Compute the Hodge Laplacian matrix of dimension 1

L1 = cx.hodge_laplacian_matrix(1)

# Compute the Hodge Laplacian matrix of dimension 2

L2 = cx.hodge_laplacian_matrix(2)
```

## Example 3: creating a combinatorial complex

```python
import toponetx as tnx

# Instantiate a combinatorial complex object with a few cells

cc = tnx.CombinatorialComplex()

# Add some cells of different ranks after initialization

cc.add_cell([1, 2, 3], rank=2)
cc.add_cell([3, 4, 5], rank=2)
cc.add_cells_from([[2, 3, 4, 5], [3, 4, 5, 6, 7]], ranks=3)

# Compute the incidence matrix between cells of rank 0 and 2

B02 = cc.incidence_matrix(0, 2)

# Compute the incidence matrix between cells of rank 0 and 3

B03 = cc.incidence_matrix(0, 3)
```

## 🔍 References ##

To learn more about how topological domains are used in deep learning:

- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzmán-Sáenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K. Dey, Soham Mukherjee, Shreyas N. Samaga, Neal Livesay, Robin Walters, Paul Rosen, Michael T. Schaub. [Topological Deep Learning: Going Beyond Graph Data](https://arxiv.org/abs/2206.00606).
```
@misc{hajij2023topological,
      title={Topological Deep Learning: Going Beyond Graph Data},
      author={Mustafa Hajij and Ghada Zamzmi and Theodore Papamarkou and Nina Miolane and Aldo Guzmán-Sáenz and Karthikeyan Natesan Ramamurthy and Tolga Birdal and Tamal K. Dey and Soham Mukherjee and Shreyas N. Samaga and Neal Livesay and Robin Walters and Paul Rosen and Michael T. Schaub},
      year={2023},
      eprint={2206.00606},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
- Mathilde Papillon, Sophia Sanborn, Mustafa Hajij, Nina Miolane. [Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.](https://arxiv.org/pdf/2304.10031.pdf)
```
@misc{papillon2023architectures,
      title={Architectures of Topological Deep Learning: A Survey on Topological Neural Networks},
      author={Mathilde Papillon and Sophia Sanborn and Mustafa Hajij and Nina Miolane},
      year={2023},
      eprint={2304.10031},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# ⭐ Acknowledgements

`TopoNetX` has been built with the help of several open-source packages.
All of these are listed in setup.py.
Some of these packages include:
- [`NetworkX`](https://networkx.org/)
- [`HyperNetX`](https://pnnl.github.io/HyperNetX/build/index.html)
- [`gudhi`](https://gudhi.inria.fr/python/latest/)
- [`trimesh`](https://trimsh.org/index.html)
