[![Test](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/TopoNetX/actions/workflows/lint.yml)

TopoNetX
=========

The TNX library provides classes and methods for modeling the entities and relationships
found in higher order networks such as simplicial, cellular, CW and combinatorial complexes.
This library serves as a repository of the methods and alxgorithms we find most useful
as we explore what higher order networks can tell us .

TNX was developed by pyt-team.


New Features of Version 1.0
---------------------------


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