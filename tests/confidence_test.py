import pytest
import yaml
from pathlib import Path

@pytest.fixture(scope="session")
def testcfg():
    with open("configs/tests.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def videos_dir(testcfg):
    return Path(testcfg["videos_dir"])

@pytest.fixture(scope="session")
def fixtures_dir(testcfg):
    return Path(testcfg["fixtures_dir"])
