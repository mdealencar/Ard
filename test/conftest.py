import glob
import os
import shutil
import pytest


def pytest_sessionfinish(session, exitstatus):
    # Cleanup code after tests
    pytest_out_dir_pattern = "pytest*_out"
    for pytest_out_dir in glob.glob(pytest_out_dir_pattern):
        if os.path.exists(pytest_out_dir):
            shutil.rmtree(pytest_out_dir)
