"""Load Austin R. Benson's datasets.

This module provides functions to load the datasets by Austin R. Benson, published on
his website:

    https://www.cs.cornell.edu/~arb/data/

A dataset is stored in a folder with the following mandatory files:

- ``{name}-nverts.txt``: Line ``i`` contains the number of vertices in the ``i``-th simplex.

- ``{name}-simplices.txt``: Each line contains a vertex ID. Line 1 corresponds to the
  first vertex of the first simplex, line 2 to the second vertex of the first simplex,
  and so on.

Optionally, additional files can be present and are consumed by the functions in this
module:

- ``{name}-times.txt``: Line ``i`` contains the timestamp of the ``i``-th simplex.
"""

from pathlib import Path

from more_itertools import spy

from toponetx.classes.simplex import Simplex

__all__ = ["load_benson_hyperedges", "load_benson_simplices"]


def _validate_folder(folder: Path) -> str:
    """Validate that the given folder contains a valid Benson dataset.

    Parameters
    ----------
    folder : Path
        Path to the folder containing the dataset.

    Returns
    -------
    str
        The name of the dataset (the folder name).

    Raises
    ------
    ValueError
        If the folder does not exist or is not in the expected format.
    """
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder `{folder}` does not exist.")

    name = folder.name

    mandatory_files = [f"{name}-nverts.txt", f"{name}-simplices.txt"]
    if not all((folder / file).exists() for file in mandatory_files):
        raise ValueError(f"Folder `{folder}` is not in an expected format.")

    return name


def load_benson_hyperedges(folder: Path | str) -> tuple[list[Simplex], list[Simplex]]:
    """Load hyperedge data from the Benson dataset format.

    Parameters
    ----------
    folder : Path | str
        Path to the folder containing the dataset.

    Returns
    -------
    nodes : list[Simplex]
        List of nodes in the dataset, each represented as a `Simplex` with a single
        vertex and associated name and label.
    simplices : list[Simplex]
        List of hyperedges in the dataset.

    Raises
    ------
    ValueError
        If the folder does not exist or is not in the expected format.
    """
    if not isinstance(folder, Path):
        folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder `{folder}` does not exist.")

    name = folder.name

    if not all(
        (folder / file).exists()
        for file in [f"label-names-{name}.txt", f"node-labels-{name}.txt"]
    ):
        raise ValueError(f"Folder `{folder}` is not in an expected format.")

    with (folder / f"label-names-{name}.txt").open() as file:
        label_map = {index: line.strip() for index, line in enumerate(file, start=1)}
    with (folder / f"node-labels-{name}.txt").open() as file:
        # Peek at the first 10 lines to check for commas (multilabel) or not (single label)
        first_lines, file_iter = spy(file, 10)
        is_multilabel = any("," in line for line in first_lines)

        if is_multilabel:
            node_labels = [
                [label_map[int(idx)] for idx in line.strip().split(",")]
                for line in file_iter
            ]
        else:
            node_labels = [label_map[int(line.strip())] for line in file_iter]

    nodes = [
        Simplex([vertex], label=label)
        for vertex, label in enumerate(node_labels, start=1)
    ]

    if (folder / f"node-names-{name}.txt").exists():
        with (folder / f"node-names-{name}.txt").open() as file:
            for simplex, node_name in zip(nodes, file, strict=True):
                simplex["name"] = node_name.strip()

    with (folder / f"hyperedges-{name}.txt").open() as file:
        simplices = [
            Simplex([int(vertex) for vertex in line.split(",")]) for line in file
        ]

    return nodes, simplices


def load_benson_simplices(folder: Path | str) -> list[Simplex]:
    """Load simplicial complex data from the Benson dataset format.

    If the dataset is temporal (indicated by the presence of a ``{name}-times.txt``
    file), the simplices are guaranteed to be in chronological order.

    Parameters
    ----------
    folder : Path | str
        Path to the folder containing the dataset.

    Returns
    -------
    list[Simplex]
        List of simplices in the dataset.

    Raises
    ------
    ValueError
        If the folder does not exist or is not in the expected format.

    Notes
    -----
    - If the dataset has node labels, simplices have a ``label`` attribute.
    - If the dataset is temporal, simplices have a ``time`` attribute.
    """
    if not isinstance(folder, Path):
        folder = Path(folder)

    name = _validate_folder(folder)

    with (
        (folder / f"{name}-nverts.txt").open() as num_vertices_file,
        (folder / f"{name}-simplices.txt").open() as simplices_file,
    ):
        simplices = [
            Simplex([int(simplices_file.readline()) for _ in range(simplex_size)])
            for simplex_size in map(int, num_vertices_file)
        ]

    if (folder / f"{name}-times.txt").exists():
        with (folder / f"{name}-times.txt").open() as times_file:
            simplex_times = [int(line) for line in times_file]
        for time, simplex in zip(simplex_times, simplices, strict=True):
            simplex["time"] = time
        simplices = sorted(simplices, key=lambda simplex: simplex["time"])

    return simplices
