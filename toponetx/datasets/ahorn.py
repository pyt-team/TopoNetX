"""Module to load datasets from the Aachen Higher-Order Repository of Networks.

This module provides functions to load and read higher-order network datasets from the
Aachen Higher-Order Repository of Networks (AHORN). Datasets can be loaded directly from
the remote repository or read from local files.

To use any function from this module, the optional ``ahorn-loader`` package must be
installed.

References
----------
https://ahorn.rwth-aachen.de/

Examples
--------
Load a dataset directly from AHORN:

>>> import toponetx as tnx
>>> SC = tnx.datasets.load_ahorn_dataset("dataset_name")

Read a dataset from a local file:

>>> import toponetx as tnx
>>> CC = tnx.datasets.read_ahorn_dataset(
...     "path/to/dataset.txt", create_using=tnx.CellComplex
... )
"""

import json
from pathlib import Path
from typing import IO

from toponetx.classes import CellComplex, SimplicialComplex
from toponetx.classes.complex import Complex

try:
    import ahorn_loader
except ImportError:
    ahorn_loader = None


__all__ = ["load_ahorn_dataset", "read_ahorn_dataset"]


def _assert_ahorn_loader_installed() -> None:
    """Ensure that the ``ahorn-loader`` package is installed.

    Raises
    ------
    RuntimeError
        If the ``ahorn-loader`` package is not installed.
    """
    if ahorn_loader is None:
        raise RuntimeError(
            "To load datasets from AHORN, please install the optional `ahorn-loader` package."
        )


def load_ahorn_dataset[T: Complex](name: str, create_using: type[T] | None = None) -> T:
    """Load the specified dataset from the Aachen Higher-Order Repository of Networks.

    The dataset file will be stored in your system cache and can be deleted according
    to your system's cache policy. To ensure that costly re-downloads do not occur, use
    ``ahorn_loader.download_dataset`` function to store the dataset file at a more
    permanent location and then use ``read_ahorn_dataset`` to read from that location.

    Parameters
    ----------
    name : str
        The name of the dataset to load.
    create_using : type[Complex], default=SimplicialComplex
        The type of complex to create.

    Returns
    -------
    Complex
        The complex representing the AHORN dataset.

    Raises
    ------
    ValueError
        If the dataset name is invalid or not found.
    RuntimeError
        If the ``ahorn-loader`` package is not installed.
    RuntimeError
        If the dataset could not be downloaded.
    """
    _assert_ahorn_loader_installed()

    try:
        with ahorn_loader.read_dataset(name) as dataset:
            return _read_ahorn_dataset(dataset, create_using=create_using)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{name}': {e!s}") from e


def read_ahorn_dataset[T](
    path: str | Path | IO[str], create_using: type[T] | None = None
) -> T:
    """Read an AHORN dataset from a local file or file-like object.

    This function accepts file paths and file-like objects provided by users. When
    working with paths from untrusted sources, users should validate and sanitize the
    input paths to prevent potential path traversal attacks. This is important when
    accepting user-provided file paths in production environments.

    Parameters
    ----------
    path : str or Path or IO
        The path to the AHORN dataset file or an open file descriptor.
    create_using : type[Complex], default=SimplicialComplex
        The type of complex to create.

    Returns
    -------
    Complex
        The complex representing the AHORN dataset.

    Raises
    ------
    RuntimeError
        If the ``ahorn-loader`` package is not installed.
    RuntimeError
        If the dataset could not be read.
    """
    _assert_ahorn_loader_installed()

    try:
        if isinstance(path, str):
            path = Path(path)

        if isinstance(path, Path):
            with path.open() as f:
                return _read_ahorn_dataset(f, create_using=create_using)
        else:
            return _read_ahorn_dataset(path, create_using=create_using)
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset: {e!s}") from e


def _read_ahorn_dataset[T](file: IO[str], create_using: type[T] | None = None) -> T:
    """Read AHORN dataset from file-like object.

    Parameters
    ----------
    file : IO
        An open file descriptor to read the AHORN dataset from.
    create_using : type[Complex], default=SimplicialComplex
        The type of complex to create.

    Returns
    -------
    Complex
        The complex representing the AHORN dataset.
    """
    if create_using is None:
        create_using = SimplicialComplex

    complex_obj = create_using(**json.loads(next(file)))

    for line_num, line in enumerate(file, start=2):
        try:
            elements_part, metadata = line.split(" ", maxsplit=1)
            elements = list(map(int, elements_part.split(",")))
        except ValueError as e:
            raise ValueError(
                f"Malformed AHORN dataset at line {line_num}: expected space-separated "
                f"elements and metadata, but got: {line.strip()}"
            ) from e

        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Malformed metadata at line {line_num}: expected valid JSON, "
                f"but got: {metadata}"
            ) from e

        if len(elements) == 1:
            complex_obj.add_node(elements[0], **metadata)
        else:
            # Use appropriate method based on complex type
            if isinstance(complex_obj, SimplicialComplex):
                complex_obj.add_simplex(elements, **metadata)
            elif isinstance(complex_obj, CellComplex):
                complex_obj.add_cell(elements, rank=len(elements) - 1, **metadata)
            else:
                raise TypeError(f"Unsupported complex type: {type(complex_obj)}")

    return complex_obj
