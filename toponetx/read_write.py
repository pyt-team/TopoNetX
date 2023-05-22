"""Read/write utilities."""

import pickle


def to_pickle(obj, filename):
    """Write object to a pickle file."""
    with open(f"{filename}", "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(filepath):
    """Return object from file."""
    with open(filepath, "rb") as f:
        temp = pickle.load(f)
    return temp
