import glob
from pathlib import Path
import shutil
import pytest


def pytest_sessionfinish(session, exitstatus):
    # Cleanup code after tests
    pytest_out_dir_pattern = "pytest*_out"
    for pytest_out_dir in glob.glob(pytest_out_dir_pattern):
        if Path(pytest_out_dir).exists:
            shutil.rmtree(pytest_out_dir)
