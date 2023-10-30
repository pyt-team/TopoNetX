"""Unit tests for the tutorials."""

import glob
import subprocess
import tempfile

import pytest


def _exec_tutorial(path):
    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1000",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output",
        file_name,
        path,
    ]
    subprocess.check_call(args)


TUTORIALS_DIR = "tutorials"
paths = sorted(glob.glob(f"{TUTORIALS_DIR}/*.ipynb"))


@pytest.mark.parametrize("path", paths)
def test_tutorial(path):
    """Test the tutorials."""
    _exec_tutorial(path)
