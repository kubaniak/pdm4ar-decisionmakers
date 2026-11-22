"""SymPy dynamics (spaceship.py) vs. the simulator model and vs. the vectorised numpy implementation."""
import numpy as np
import pytest
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState

from pdm4ar.exercises.ex13.discretization import FirstOrderHold
from pdm4ar.exercises.ex13.spaceship import SpaceshipDyn
from pdm4ar.exercises_def.ex13.disturbance import FuelSpaceshipModel


@pytest.fixture(scope="module")
def dyn(sg, sp):
    d = SpaceshipDyn(sg, sp)
    return d, d.get_dynamics()


def random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.uniform(-10, 10, n),
            rng.uniform(-10, 10, n),
            rng.uniform(-4, 4, n),
            rng.uniform(-2.5, 2.5, n),
            rng.uniform(-2.5, 2.5, n),
            rng.uniform(-1.5, 1.5, n),
            rng.uniform(-1.0, 1.0, n),
            rng.uniform(2.05, 2.5, n),
        ]
    )
    U = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-0.78, 0.78, n)])
    return X, U


def test_template_dynamics_check(sg, sp):
    """The check provided with the template (comparison with FuelSpaceshipModel)."""
    assert FirstOrderHold(SpaceshipDyn(sg, sp), 5, 3).check_dynamics()


def test_dimensions(dyn):
    d, (f, A, B, F) = dyn
    assert (d.n_x, d.n_u, d.n_p) == (8, 2, 1)
    x, u, p = np.ones(8) * 0.3 + np.eye(8)[7] * 2, np.array([1.0, 0.1]), np.array([5.0])
    assert np.asarray(f(x, u, p)).shape == (8, 1)
    assert np.asarray(A(x, u, p)).shape == (8, 8)
    assert np.asarray(B(x, u, p)).shape == (8, 2)
    assert np.asarray(F(x, u, p)).shape == (8, 1)


@pytest.mark.parametrize("seed", range(3))
def test_matches_simulator_model_and_time_scaling(dyn, sg, sp, seed):
    _, (f, *_rest) = dyn
    X, U = random_points(20, seed)
    for x, u in zip(X, U):
        # stay inside the actuator limits so the simulator does not clip
        u = np.clip(u, [sp.thrust_limits[0], sp.ddelta_limits[0]], [sp.thrust_limits[1], sp.ddelta_limits[1]])
        x[6] = np.clip(x[6], *sp.delta_limits) * 0.99
        state, cmd = SpaceshipState(*x), SpaceshipCommands(*u)
        expected = FuelSpaceshipModel(state, sg, sp).dynamics(state, cmd).as_ndarray()
        for tf in (1.0, 7.5):
            got = np.asarray(f(x, u, np.array([tf])), dtype=float).ravel()
            np.testing.assert_allclose(got, tf * expected, rtol=1e-9, atol=1e-12)


def test_mass_decreases_for_both_thrust_signs(dyn):
    _, (f, *_rest) = dyn
    x = np.array([0, 0, 0, 0, 0, 0, 0, 2.4])
    for thrust in (-1.5, 1.5):
        assert np.asarray(f(x, np.array([thrust, 0.0]), np.array([1.0]))).ravel()[7] < 0
    assert np.asarray(f(x, np.array([0.0, 0.0]), np.array([1.0]))).ravel()[7] == 0


def test_batch_matches_sympy(dyn, batch):
    _, (f, A, B, F) = dyn
    X, U = random_points(40, seed=7)
    fb, Ab, Bb = batch.jacobians_phys(X, U)
    for i, (x, u) in enumerate(zip(X, U)):
        p = np.array([3.0])
        np.testing.assert_allclose(fb[i] * 3.0, np.asarray(f(x, u, p), dtype=float).ravel(), atol=1e-12)
        np.testing.assert_allclose(Ab[i] * 3.0, np.asarray(A(x, u, p), dtype=float), atol=1e-12)
        np.testing.assert_allclose(Bb[i] * 3.0, np.asarray(B(x, u, p), dtype=float), atol=1e-12)
        np.testing.assert_allclose(fb[i], np.asarray(F(x, u, p), dtype=float).ravel(), atol=1e-12)


def test_body_frame_kinematics(batch):
    """Pure translation along the heading; Coriolis terms with rotation."""
    x = np.array([[0, 0, np.pi / 2, 1.0, 0.0, 0.0, 0.0, 2.2]])
    f = batch.f_phys(x, np.zeros((1, 2)))[0]
    np.testing.assert_allclose(f[:2], [0.0, 1.0], atol=1e-12)
    x = np.array([[0, 0, 0.0, 1.0, 0.0, 0.5, 0.0, 2.2]])
    f = batch.f_phys(x, np.zeros((1, 2)))[0]
    assert f[4] == pytest.approx(-0.5)  # dvy = -omega * vx
