import os
import sys
import tempfile
import shutil
import pytest

@pytest.fixture
def tmpdir_path():
    """Yield a temporary directory path; cleaned up after test."""
    d = tempfile.mkdtemp(prefix="washiwrap_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
