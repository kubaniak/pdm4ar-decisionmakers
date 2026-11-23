import logging
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: closed-loop simulations of whole scenarios (minutes)")


@pytest.fixture(scope="session")
def sg():
    from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry

    return SpaceshipGeometry.default()


@pytest.fixture(scope="session")
def sp():
    from dg_commons.sim.models.spaceship_structures import SpaceshipParameters

    return SpaceshipParameters.default()


@pytest.fixture(scope="session")
def batch(sg, sp):
    from pdm4ar.exercises.ex13.discretization import BatchFirstOrderHold

    return BatchFirstOrderHold(sg.l_r, sg.Iz, sp.C_T, K=30, N_sub=5)
