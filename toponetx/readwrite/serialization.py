"""Read and write complexes as pickled objects."""

import pickle
from pathlib import Path

__all__ = ["load_from_pickle", "to_pickle"]


def to_pickle(obj, filename: str) -> None:
    """Write object to a pickle file.

    Parameters
    ----------
    obj : object
        Object to write.
    filename : Path or str
        Filename.
    """
    with Path(filename).open("wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(filepath):
    """Return object from file.

    Parameters
    ----------
    filepath : Path or str
        Filepath.

    Returns
    -------
    object
        Object.
    """
    with Path(filepath).open("rb") as f:
        return pickle.load(f)  # noqa: S301
