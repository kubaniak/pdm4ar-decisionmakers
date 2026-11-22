"""Vectorised FOH discretization: sensitivities vs. finite differences, propagation vs. solve_ivp."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pdm4ar.exercises.ex13.tracker import disturbance_input_matrix


def reference(K, seed=0, tf=12.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(8, K)) * np.array([3, 3, 1, 1, 1, 0.5, 0.5, 0])[:, None]
    X[7] = 2.3 + 0.1 * rng.random(K)
    U = np.vstack([rng.uniform(0.2, 1.8, K), rng.uniform(-0.6, 0.6, K)])
    return X, U, np.array([tf])


def propagate(batch, x0, u0, u1, tf, K, dist=None):
    dt = 1.0 / (K - 1)

    def fun(t, x):
        u = u0 + (t / dt) * (u1 - u0)
        out = tf * batch.f_phys(x[None], u[None])[0]
        if dist is not None:
            out = out + tf * disturbance_input_matrix(np.array([x[2]]))[0] @ dist
        return out

    return solve_ivp(fun, (0, dt), x0, rtol=1e-11, atol=1e-12).y[:, -1]


def test_shapes(batch):
    X, U, p = reference(batch.K)
    d = batch.calculate_discretization(X, U, p, dense=True)
    n = batch.K - 1
    assert d["A"].shape == (n, 8, 8) and d["Bm"].shape == (n, 8, 2) and d["Bp"].shape == (n, 8, 2)
    assert d["F"].shape == (n, 8) and d["r"].shape == (n, 8) and d["x_next"].shape == (n, 8)
    assert d["x_dense"].shape == (n, batch.N_sub + 1, 8)
    np.testing.assert_allclose(d["x_dense"][:, 0], X[:, :-1].T)


def test_propagation_matches_solve_ivp(batch):
    X, U, p = reference(batch.K, seed=1)
    d = batch.calculate_discretization(X, U, p)
    for k in (0, 5, batch.K - 2):
        ref = propagate(batch, X[:, k], U[:, k], U[:, k + 1], p[0], batch.K)
        np.testing.assert_allclose(d["x_next"][k], ref, atol=2e-6)


def test_affine_model_exact_at_reference(batch):
    X, U, p = reference(batch.K, seed=2)
    d = batch.calculate_discretization(X, U, p)
    lin = (
        np.einsum("nij,nj->ni", d["A"], X[:, :-1].T)
        + np.einsum("nij,nj->ni", d["Bm"], U[:, :-1].T)
        + np.einsum("nij,nj->ni", d["Bp"], U[:, 1:].T)
        + d["F"] * p[0]
        + d["r"]
    )
    np.testing.assert_allclose(lin, d["x_next"], atol=1e-10)


@pytest.mark.parametrize("which", ["A", "Bm", "Bp", "F"])
def test_sensitivities_vs_finite_differences(batch, which):
    K = batch.K
    X, U, p = reference(K, seed=3)
    d = batch.calculate_discretization(X, U, p)
    k, eps = 4, 1e-6
    x0, u0, u1, tf = X[:, k], U[:, k], U[:, k + 1], p[0]
    if which == "A":
        cols = []
        for j in range(8):
            e = np.eye(8)[j] * eps
            cols.append((propagate(batch, x0 + e, u0, u1, tf, K) - propagate(batch, x0 - e, u0, u1, tf, K)) / (2 * eps))
        fd = np.column_stack(cols)
    elif which in ("Bm", "Bp"):
        cols = []
        for j in range(2):
            e = np.eye(2)[j] * eps
            if which == "Bm":
                a, b = propagate(batch, x0, u0 + e, u1, tf, K), propagate(batch, x0, u0 - e, u1, tf, K)
            else:
                a, b = propagate(batch, x0, u0, u1 + e, tf, K), propagate(batch, x0, u0, u1 - e, tf, K)
            cols.append((a - b) / (2 * eps))
        fd = np.column_stack(cols)
    else:
        fd = (propagate(batch, x0, u0, u1, tf + eps, K) - propagate(batch, x0, u0, u1, tf - eps, K)) / (2 * eps)
    np.testing.assert_allclose(d[which][k], fd, atol=5e-6, rtol=1e-4)


def test_disturbance_propagation_and_jacobian(batch):
    K = batch.K
    X, U, p = reference(K, seed=4, tf=6.0)
    dist = np.array([0.3, -0.2, 0.05])
    d = batch.calculate_discretization(X, U, p, disturbance=(dist, 100.0))
    k = 2
    ref = propagate(batch, X[:, k], U[:, k], U[:, k + 1], p[0], K, dist)
    np.testing.assert_allclose(d["x_next"][k], ref, atol=2e-6)
    eps = 1e-6
    e = np.eye(8)[2] * eps  # psi column carries the extra disturbance terms
    fd = (propagate(batch, X[:, k] + e, U[:, k], U[:, k + 1], p[0], K, dist)
          - propagate(batch, X[:, k] - e, U[:, k], U[:, k + 1], p[0], K, dist)) / (2 * eps)
    np.testing.assert_allclose(d["A"][k][:, 2], fd, atol=5e-6)
    # inactive after the burst duration
    d_short = batch.calculate_discretization(X, U, p, disturbance=(dist, 0.5))
    d_none = batch.calculate_discretization(X, U, p)
    np.testing.assert_allclose(d_short["x_next"][-1], d_none["x_next"][-1])
    assert not np.allclose(d_short["x_next"][0], d_none["x_next"][0])


def test_simulate_zoh_matches_simulator_model(batch, sg, sp):
    from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
    from decimal import Decimal
    from pdm4ar.exercises_def.ex13.disturbance import FuelSpaceshipModel

    x0 = np.array([1.0, -2.0, 0.4, 0.5, -0.1, 0.2, 0.1, 2.4])
    u = np.array([1.3, -0.2])
    model = FuelSpaceshipModel(SpaceshipState(*x0), sg, sp)
    for _ in range(10):
        model.update(SpaceshipCommands(*u), Decimal("0.01"))
    got = batch.simulate_zoh(x0, lambda t: u, np.array([0.0, 0.1]), n_rk=8)[-1]
    np.testing.assert_allclose(got, model.get_state().as_ndarray(), atol=1e-4)
