"""TV-LQR tracker: nominal tracking, disturbance observer, feed-forward, rollouts."""
from decimal import Decimal

import numpy as np
import pytest
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState

from pdm4ar.exercises.ex13.tracker import TrackerParams, TVLQRTracker
from pdm4ar.exercises_def.ex13.disturbance import DisturbedSpaceshipModel, DynamicsDisturbanceParams
from scenario_utils import PUBLIC, load, planner_from_context


@pytest.fixture(scope="module")
def setup():
    ctx = load(PUBLIC[0])
    planner, x0, goal = planner_from_context(ctx)
    plan = planner.plan_initial(x0, goal.target, (goal.pos_tol, goal.vel_tol, goal.dir_tol))
    return planner, x0, goal, plan


def make_tracker(planner, **kw):
    sp = planner.sp
    tr = TVLQRTracker(planner.integrator, sp.thrust_limits, sp.ddelta_limits, sp.delta_limits, TrackerParams(**kw))
    return tr


def simulate(planner, tracker, plan, x0, disturbance=None, t_end=None):
    """Closed loop with the simulator's spaceship model (0.1 s commands, 0.01 s integration)."""
    sg, sp = planner.sg, planner.sp
    params = disturbance or DynamicsDisturbanceParams(start_time=100, duration=1, linear_acc_sigma=0, angular_acc_sigma=0, seed=0)
    model = DisturbedSpaceshipModel(SpaceshipState(*x0), sg, sp, params)
    t, xs, ts = 0.0, [], []
    t_end = plan.t_end if t_end is None else t_end
    while t < t_end:
        x = model.get_state().as_ndarray()
        tracker.observe(t, x)
        u = tracker.command(t, x)
        tracker.record_command(t, x, u)
        xs.append(x)
        ts.append(t)
        for _ in range(10):
            model.update(SpaceshipCommands(*u), Decimal("0.01"))
        t = round(t + 0.1, 10)
    return np.array(ts), np.array(xs), model


def test_gains_shape(setup):
    planner, x0, goal, plan = setup
    tr = make_tracker(planner)
    tr.set_plan(plan)
    assert tr.gains.shape == (len(tr.t_grid), 2, 8)
    assert tr.gains_d.shape == (len(tr.t_grid), 2, 3)
    assert np.isfinite(tr.gains).all()


def test_nominal_tracking(setup):
    planner, x0, goal, plan = setup
    tr = make_tracker(planner)
    tr.set_plan(plan)
    ts, xs, _ = simulate(planner, tr, plan, x0.as_ndarray())
    err = np.array([np.linalg.norm(x[:2] - plan.x_at(t)[:2]) for t, x in zip(ts, xs)])
    assert err.max() < 0.1
    assert np.all(tr.d_hat == 0)  # no disturbance detected in the nominal case


def test_perturbed_initial_state_converges(setup):
    planner, x0, goal, plan = setup
    tr = make_tracker(planner)
    tr.set_plan(plan)
    x_start = x0.as_ndarray() + np.array([0.3, -0.3, 0.1, 0.1, -0.1, 0, 0, 0])
    ts, xs, _ = simulate(planner, tr, plan, x_start, t_end=6.0)
    err = np.array([np.linalg.norm(x[:2] - plan.x_at(t)[:2]) for t, x in zip(ts, xs)])
    assert err[-1] < 0.1 < err[0]


def test_disturbance_observer_estimates_burst(setup):
    planner, x0, goal, plan = setup
    tr = make_tracker(planner)
    tr.set_plan(plan)
    dist = DynamicsDisturbanceParams(start_time=2.0, duration=2.0, linear_acc_sigma=0.2, angular_acc_sigma=0.04, seed=3)
    estimates = []
    sg, sp = planner.sg, planner.sp
    model = DisturbedSpaceshipModel(SpaceshipState(*x0.as_ndarray()), sg, sp, dist)
    t = 0.0
    while t < 5.0:
        x = model.get_state().as_ndarray()
        tr.observe(t, x)
        estimates.append((t, tr.d_hat.copy()))
        u = tr.command(t, x)
        tr.record_command(t, x, u)
        for _ in range(10):
            model.update(SpaceshipCommands(*u), Decimal("0.01"))
        t = round(t + 0.1, 10)
    true = np.array(model.sampled_disturbance)
    inside = [d for (tt, d) in estimates if 2.15 < tt < 3.95]
    outside = [d for (tt, d) in estimates if tt < 1.95 or tt > 4.15]
    for d in inside:
        np.testing.assert_allclose(d, true, atol=0.01)
    for d in outside:
        np.testing.assert_allclose(d, 0.0, atol=1e-9)


def test_feedforward_reduces_tracking_error(setup):
    """On average over random bursts the estimated-disturbance feed-forward reduces the peak tracking error."""
    planner, x0, goal, plan = setup
    errs = {False: [], True: []}
    for seed in range(6):
        dist = DynamicsDisturbanceParams(start_time=2.0, duration=2.5, linear_acc_sigma=0.2, angular_acc_sigma=0.04, seed=seed)
        for ff in (False, True):
            tr = make_tracker(planner, use_disturbance_ff=ff)
            tr.set_plan(plan)
            ts, xs, _ = simulate(planner, tr, plan, x0.as_ndarray(), disturbance=dist, t_end=7.0)
            errs[ff].append(max(np.linalg.norm(x[:2] - plan.x_at(t)[:2]) for t, x in zip(ts, xs)))
    assert np.mean(errs[True]) < np.mean(errs[False])


def test_rollout_matches_closed_loop_simulation(setup):
    planner, x0, goal, plan = setup
    tr = make_tracker(planner)
    tr.set_plan(plan)
    x_start = x0.as_ndarray() + np.array([0.2, 0.1, 0.05, 0, 0, 0, 0, 0])
    ts, xs, us = tr.rollout(x_start, 0.0, 5.0, n_rk=4)
    tr2 = make_tracker(planner)
    tr2.set_plan(plan)
    ts2, xs2, _ = simulate(planner, tr2, plan, x_start, t_end=5.0)
    n = min(len(xs), len(xs2))
    assert np.max(np.abs(xs[:n, :2] - xs2[:n, :2])) < 1e-2


def test_commands_within_limits(setup):
    planner, x0, goal, plan = setup
    tr = make_tracker(planner)
    tr.set_plan(plan)
    sp = planner.sp
    x = plan.x_at(3.0) + np.array([3.0, -3.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    tr.d_hat = np.array([1.0, -1.0, 0.3])
    u = tr.command(3.0, x)
    assert sp.thrust_limits[0] <= u[0] <= sp.thrust_limits[1]
    assert sp.ddelta_limits[0] <= u[1] <= sp.ddelta_limits[1]
