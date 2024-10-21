<h2 align="center">
  <img src="https://raw.githubusercontent.com/pyt-team/TopoNetX/main/resources/logo.png" height="250">
</h2>

<h3 align="center">
   Computing with Relational Data Abstracted as Topological Domains
</h3>

<p align="center">
  <a href="#-scope-and-functionality">Scope and Functionality</a> •
  <a href="#%EF%B8%8F-main-features">Main Features</a> •
  <a href="#-installing-toponetx">Installing TopoNetX</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-references">References</a> •
  <a href="#-acknowledgements">Acknowledgements</a>
</p>

<div align="center">

[![Test](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/pyt-team/TopoNetX/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pyt-team/TopoNetX)
[![Docs](https://img.shields.io/badge/docs-website-brightgreen)](https://pyt-team.github.io/toponetx/index.html)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![license](https://badgen.net/github/license/pyt-team/TopoNetX?color=green)](https://github.com/pyt-team/TopoNetX/blob/main/LICENSE)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/pyt-teamworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7958504.svg)](https://doi.org/10.5281/zenodo.7958504)

</div>

![toponetx](https://user-images.githubusercontent.com/8267869/234068354-af9480f1-1d18-4914-92f1-916d9093e44d.png)

Many complex systems, ranging from socio-economic systems such as social networks, over to biological systems (e.g., proteins) and technical systems can be abstracted as a set of entities with are linked to each other via a set of relations.
For instance, a social network may be abstracted as a set vertices corresponding to people linked via various social interactions, including pairwise relationships such as friendships and higher-order relationships involving multiple people.
This _relational data_ can be abstracted as a topological domain such as a graph, hypergraph, simplicial, cellular path or combinatorial complex, which enables the principled analysis of such data.

`TopoNetX` provides a unified platform to compute with such relational data.

## 🎯 Scope and functionality

`TopoNetX` (TNX) is a package for computing with topological domains and studying their properties.

With its dynamic construction capabilities and support for arbitrary
attributes and data, `TopoNetX` allows users to easily explore the topological structure
of their data and gain insights into its underlying geometric and algebraic properties.

Available functionality ranges
from computing boundary operators and Hodge Laplacians on simplicial/cell/combinatorial complexes
to performing higher-order adjacency calculations.

TNX is similar to [`NetworkX`](https://networkx.org/), a popular graph package, and extends its capabilities to support a
wider range of mathematical structures, including cell complexes, simplicial complexes and
combinatorial complexes.
The TNX library provides classes and methods for modeling the entities and relations
found in higher-order networks such as simplicial, cellular, CW and combinatorial complexes.
This package serves as a repository of the methods and algorithms we find most useful
as we explore the knowledge that can be encoded via higher-order networks.

TNX supports the construction of many topological structures including the `CellComplex`, `PathComplex`, "ColoredHyperGraph" `SimplicialComplex` and `CombinatorialComplex` classes.
These classes provide methods for computing boundary operators, Hodge Laplacians
and higher-order adjacency operators on cell, simplicial and combinatorial complexes,
respectively. The classes are used in many areas of mathematics and computer science,
such as algebraic topology, geometry, and data analysis.

TNX is developed by the [pyt-team](https://github.com/pyt-team)

## 🛠️ Main features

1. Dynamic construction of cell, simplicial and combinatorial complexes, allowing users to add or remove objects from these structures after their initial creation.
2. Compatibility with the [`NetworkX`](https://networkx.org/) and [`gudhi`](https://gudhi.inria.fr/) packages, enabling users to
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

`TopoNetX` is available on PyPI and can be installed using `pip`:

```bash
pip install toponetx
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

## 🧑‍💻 Install from source

To install the latest version from source, follow these steps:

1. Clone a copy of `TopoNetX` from source:

```bash
git clone https://github.com/pyt-team/TopoNetX
cd TopoNetX
```

2. If you have already cloned `TopoNetX` from source, update it:

```bash
git pull
```

3. Install `TopoNetX` in editable mode (requires `pip` ≥ 21.3 for [PEP 660](https://peps.python.org/pep-0610/) support):

```bash
pip install -e '.[all]'
```

4. Install pre-commit hooks:

```bash
pre-commit install
```

## 🔍 References

TopoNetX is a part of TopoX, a suite of Python packages for machine learning on topological domains. If you find TopoNetX useful please consider citing our software paper:

- Hajij et al. 2023. [TopoX: a suite of Python packages for machine learning on topological domains](https://arxiv.org/abs/2402.02441)

```
@article{hajij2024topox,
  title={TopoX: A Suite of Python Packages for Machine Learning on Topological Domains},
  author={PYT-Team},
  journal={arXiv preprint arXiv:2402.02441},
  year={2024}
}
```

To learn more about topological domains, and how they can be used in deep learning:

- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzmán-Sáenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K. Dey, Soham Mukherjee, Shreyas N. Samaga, Neal Livesay, Robin Walters, Paul Rosen, Michael T. Schaub.   [Topological Deep Learning: Going Beyond Graph Data](https://arxiv.org/abs/2206.00606) (arXiv) • [Topological Deep Learning: A Book](https://tdlbook.org/)

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
- [`HyperNetX`](https://pnnl.github.io/HyperNetX/)
- [`gudhi`](https://gudhi.inria.fr/python/latest/)
- [`trimesh`](https://trimsh.org/index.html)

## Funding

<img align="right" width="200" src="https://raw.githubusercontent.com/pyt-team/TopoNetX/main/resources/erc_logo.png">

Partially funded by the European Union (ERC, HIGH-HOPeS, 101039827). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

Partially funded by the National Science Foundation (DMS-2134231, DMS-2134241).
