"""Ship circle cover, analytic obstacle clearances and their gradients, exact collision checks."""
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon

from pdm4ar.exercises.ex13.geometry import (
    ExactCollisionChecker,
    ObstacleSet,
    boundary_halfplanes,
    dock_segments,
    polygon_contains_ship,
    ship_circles,
    ship_footprints,
    wrap_angle,
)
from pdm4ar.exercises_def.ex13.goal import DockingTarget
from dg_commons.sim.models.obstacles_dyn import DynObstacleState


@pytest.fixture(scope="module")
def circles(sg):
    return ship_circles(sg)


def test_cover_contains_ship(sg, circles):
    assert polygon_contains_ship(circles, sg)
    assert circles.n >= 3
    # the cover is not wildly conservative laterally (dock clearance is ~0.55 m per side)
    assert circles.radius <= 0.55


def test_wrap_angle():
    np.testing.assert_allclose(wrap_angle([0.0, np.pi + 0.1, -np.pi - 0.1, 4 * np.pi]), [0, -np.pi + 0.1, np.pi - 0.1, 0], atol=1e-12)


def test_footprints_match_dg_commons(sg):
    from dg_commons.sim.models.spaceship import SpaceshipState
    from pdm4ar.exercises_def.ex13.disturbance import FuelSpaceshipModel
    from dg_commons.sim.models.spaceship_structures import SpaceshipParameters

    x = np.array([[1.0, -2.0, 0.7, 0, 0, 0, 0, 2.5]])
    poly = ship_footprints(sg, x)[0]
    ref = FuelSpaceshipModel(SpaceshipState(*x[0]), sg, SpaceshipParameters.default()).get_footprint()
    assert poly.symmetric_difference(ref).area < 1e-9


def test_boundary_halfplanes_square():
    corners = np.array([[-11, -11], [-11, 11], [11, 11], [11, -11], [-11, -11]])
    n, w = boundary_halfplanes(corners)
    assert len(n) == 4
    # origin is inside: n.q <= w
    assert np.all(n @ np.zeros(2) <= w)
    assert np.all(np.isclose(w, 11.0))
    # a point outside violates exactly one half-plane
    assert np.sum(n @ np.array([12.0, 0.0]) > w) == 1


def test_dock_segments_inside_plottable_geometry():
    goal = DockingTarget(
        target=DynObstacleState(x=8.5, y=0.0, psi=3.14159, vx=0, vy=0, dpsi=0),
        pos_tol=0.5, vel_tol=0.2, dir_tol=0.1, add_land_space=0.5, arms_length=1.3, offset=1.8,
    )
    geom = goal.get_plottable_geometry()
    a, b = dock_segments(goal)
    for ai, bi in zip(a, b):
        assert geom.buffer(1e-6).contains(LineString([ai, bi]))
    assert abs(geom.area - sum(0.1 * np.linalg.norm(bi - ai) for ai, bi in zip(a, b))) < 0.05


def make_obstacles():
    obs = ObstacleSet()
    obs.circ_c0 = np.array([[0.0, 0.0], [5.0, 5.0]])
    obs.circ_v = np.array([[0.0, 0.0], [-1.0, 0.5]])
    obs.circ_r = np.array([2.0, 1.0])
    obs.seg_a = np.array([[-5.0, 3.0]])
    obs.seg_b = np.array([[-3.0, 6.0]])
    obs.seg_r = np.array([0.1])
    obs.hp_n, obs.hp_w = boundary_halfplanes(np.array([[-11, -11], [-11, 11], [11, 11], [11, -11]]))
    obs.hp_r = np.full(4, 0.1)
    return obs


def test_clearance_values(circles):
    obs = make_obstacles()
    X = np.zeros((8, 1))
    X[0, 0], X[1, 0] = 4.0, 0.0  # psi = 0: circles at x = 4 + offsets
    g, grad, gp = obs.evaluate(X, p=1.0, t0=0.0, tau=np.array([0.0]), ship=circles)
    assert g.shape == (obs.n_obs * circles.n, 1)
    # planet: first block of n_c rows
    expected = 4.0 + circles.offsets - 2.0 - circles.radius
    np.testing.assert_allclose(g[: circles.n, 0], expected)
    # right wall (x <= 11 - 0.1 - r)
    assert np.min(g[-4 * circles.n :]) == pytest.approx(11 - 0.1 - circles.radius - (4 + circles.offsets.max()))


def test_gradients_vs_finite_differences(circles):
    obs = make_obstacles()
    rng = np.random.default_rng(0)
    K = 6
    X = np.zeros((8, K))
    X[0] = rng.uniform(-6, 6, K)
    X[1] = rng.uniform(-6, 6, K)
    X[2] = rng.uniform(-3, 3, K)
    tau = np.linspace(0, 1, K)
    p, t0 = 7.0, 1.5
    g, grad, gp = obs.evaluate(X, p, t0, tau, circles)
    eps = 1e-6
    for j in range(3):
        Xp, Xm = X.copy(), X.copy()
        Xp[j] += eps
        Xm[j] -= eps
        fd = (obs.evaluate(Xp, p, t0, tau, circles)[0] - obs.evaluate(Xm, p, t0, tau, circles)[0]) / (2 * eps)
        np.testing.assert_allclose(grad[..., j], fd, atol=1e-5)
    fd_p = (obs.evaluate(X, p + eps, t0, tau, circles)[0] - obs.evaluate(X, p - eps, t0, tau, circles)[0]) / (2 * eps)
    np.testing.assert_allclose(gp, fd_p, atol=1e-5)


def test_analytic_clearance_is_conservative(sg, circles):
    """g >= 0 for all circles implies no exact collision (cover contains the ship)."""
    obs = ObstacleSet()
    obs.circ_c0 = np.array([[0.0, 0.0]])
    obs.circ_v = np.zeros((1, 2))
    obs.circ_r = np.array([2.0])
    checker = ExactCollisionChecker(sg, [Point(0, 0).buffer(2.0)], np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    rng = np.random.default_rng(1)
    X = np.zeros((8, 400))
    X[0] = rng.uniform(-5, 5, 400)
    X[1] = rng.uniform(-5, 5, 400)
    X[2] = rng.uniform(-np.pi, np.pi, 400)
    g, _, _ = obs.evaluate(X, 1.0, 0.0, np.zeros(400), circles)
    safe = np.all(g >= 0, axis=0)
    exact = checker.clearance(X.T, np.zeros(400))
    assert safe.sum() > 50
    assert np.all(exact[safe] > -1e-9)


def test_exact_checker_moving_asteroid(sg):
    checker = ExactCollisionChecker(
        sg, [], asteroid_c0=np.array([[10.0, 0.0]]), asteroid_v=np.array([[-1.0, 0.0]]), asteroid_r=np.array([1.0])
    )
    states = np.zeros((2, 8))
    d = checker.clearance(states, np.array([0.0, 10.0]))
    assert d[0] == pytest.approx(10.0 - 1.0 - sg.l_f - sg.l_c, abs=1e-6)  # nose tip points at it
    assert d[1] < 0  # asteroid reached the ship
