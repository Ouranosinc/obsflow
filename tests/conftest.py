import pytest
import xscen as xs
from xscen.config import CONFIG
from pathlib import Path

xs.load_config("../paths_obs.yml",
               "../config_obs.yml",
               verbose=(__name__ == "__main__"),
               reset=True
               )


@pytest.fixture
def pcat():
    return xs.ProjectCatalog(CONFIG["paths"]["project_catalog"])


@pytest.fixture
def ref_path():
    return Path(CONFIG["paths"]["test_reference"])


@pytest.fixture
def bbox():
    return {'lat_bnds': [47, 50], 'lon_bnds': [-75, -72]}
