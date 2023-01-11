[![Test](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml)

TopoNetX
=========


TopoNetX is a powerful and versatile package for studying the topological properties
of shapes and spaces. With its dynamic construction capabilities and support for arbitrary
attributes and data, TopoNetX allows users to easily explore the topological structure
of their data and gain insights into its underlying geometric and algebraic properties.
From computing boundary operators and Hodge Laplacians on simplicial complexes,
to performing higher-order adjacency calculations.

TopoNetX is similar to NetworkX, a popular graph package, but extends its capabilities to support a
wider range of mathematical structures, including cell complexes, simplicial complexes, and
combinatorial complexes. The TNX library provides classes and methods for modeling the entities and relationships
found in higher order networks such as simplicial, cellular, CW and combinatorial complexes.
This library serves as a repository of the methods and algorithms we find most useful
as we explore what higher order networks can tell us.


The package supports the construction many topological structures including the CellComplex, SimplicialComplex, and CombinatorialComplex classes.
 The classes provide methods for computing boundary operators, Hodge Laplacians,
 and higher-order adjacency operators on the cell, simplicial, and combinatorial complexes,
  respectively. The classes are used in many areas of mathematics and computer science,
  such as algebraic topology, geometry, and data analysis.



TNX was developed by pyt-team.




New Features of Version 1.0
---------------------------

1. Dynamic construction of cell complexes, simplicial complexes, and combinatorial complexes, allowing users to
 add or remove objects from these structures after their initial creation.
2. Compatibility with the NetworkX and gudhi libraries, enabling users to
    leverage the powerful algorithms and data structures provided by these packages.
3. Support for attaching arbitrary attributes and data to cells, simplices, and other entities in the complex, allowing users to store and manipulate additional information about these objects.
4. Computation of boundary operators, Hodge Laplacians, and higher-order adjacency
     operators on the complex, enabling users to study the topological properties of the space.
6. Robust error handling and validation of input data, ensuring that the package is
    reliable and easy to use.
7. Support of higher order represenation learning algorithsm such as DeepCell, Cell2Vec, Higher Order Laplacian Eigenmaps and Higher Order Geometric Laplacian Eigenmaps for various complexes supported in TopoNetX (simplicial, cellular, combinatorial).


Installing TopoNetX
====================

1. Clone a copy of TopoNetX from source:

   ```bash
   git clone https://github.com/pyt-team/TopoNetX
   cd toponetx
   ```

2. If you already cloned TopoNetX from source, update it:

   ```bash
   git pull
   ```

3. Install TopoNetX in editable mode:

   ```bash
   pip install -e ".[dev,full]"
   ```

4. Install pre-commit hooks:

   ```bash
    pre-commit install
   ```
   

## Getting Started : creating a simplicial complex 

```ruby
import toponetx as tnx

# create a SimplicialComplex object with a few simplices
sc = tnx.SimplicialComplex([[1, 2, 3], [2, 3, 4], [0, 1]])

# compute the incidence matrices 

B1 = sc.incidence_matrix(1)

B2 = sc.incidence_matrix(2)

```


## Getting Started : creating a cell complex 

```ruby
import toponetx as tnx

# create a CellComplex object with a few cells

cx = tnx.CellComplex([[1, 2, 3, 4], [3,4,5,6,7,8]],ranks=2)

# adding edges, or ranked 1 cells, can be done using the command 
cx.add_edge(0,1)

# compute the Hodge Laplacian matrices 

L1 = cx.hodge_laplacian_matrix(1)

L2 = cx.hodge_laplacian_matrix(2)
```

## Getting Started : creating a combinatorial complex 

```ruby
import toponetx as tnx

# create a combinatorial complex object with a few cells
cc = tnx.CombinatorialComplex()

# adding a cell

cc.add_cell([1, 2, 3], rank=2)
cc.add_cell([3, 4, 5], rank=2)
cc.add_cells_from([[2, 3, 4, 5], [3, 4, 5, 6, 7]], ranks=3)

# incidence between 0 and 2 cells
B02 = cc.incidence_matrix(0,2) 

# incidence between 0 and 3 cells

B03 = cc.incidence_matrix(0,3)
```

## Getting Started : simplicia/cellular/combinatorial representation learning

```ruby
import toponetx as tnx

# create a cell complex object with a few cells
cx = tnx.CellComplex([[1, 2, 3, 4], [3,4,5,6,7,8]],ranks=2)

# create a model

model = tnx.Cell2Vec()

# fit the model

model.fit(cx,neighborhood_type="adj", neighborhood_dim={"r": 1, "k": -1})
# here neighborhood_dim={"r": 1, "k": -1} specifies the dimension for
# which the cell embeddings are going to be computed. 
# r=1 means that the embeddings will be computed for the first dimension.
# The integer 'k' is ignored and only considered
# when the input complex is a combinatorial complex.


# get the embeddings:

embeddings = model.get_embedding() 

```


## Acknowledgements

TopoNetX is built with the help of several open source packages. All of these are listed in setup.py. Some of these packages include:

- NetworkX https://networkx.org/

- HyperNetX https://pnnl.github.io/HyperNetX/build/index.html

   
