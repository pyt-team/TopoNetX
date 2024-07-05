"""Read and write complexes as pickled objects."""

import pickle

__all__ = ["to_pickle", "load_from_pickle"]


def to_pickle(obj, filename: str) -> None:
    """Write object to a pickle file.

    Parameters
    ----------
    obj : object
        Object to write.
    filename : str
        Filename.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(filepath):
    """Return object from file.

    Parameters
    ----------
    filepath : str
        Filepath.

    Returns
    -------
    object
        Object.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)
