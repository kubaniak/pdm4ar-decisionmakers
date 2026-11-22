"""SCvx planner on the public scenarios: feasibility, constraints, terminal set, replanning."""
import numpy as np
import pytest

from scenario_utils import PUBLIC, load, planner_from_context


@pytest.fixture(scope="module", params=[1, 2, 3, 4], ids=lambda i: f"public{i}")
def planned(request):
    ctx = load(PUBLIC[request.param - 1])
    planner, x0, goal = planner_from_context(ctx)
    tols = (goal.pos_tol, goal.vel_tol, goal.dir_tol)
    plan = planner.plan_initial(x0, goal.target, tols)
    return planner, x0, goal, tols, plan


def test_initial_plan_feasible(planned):
    planner, x0, goal, tols, plan = planned
    assert plan.feasible and plan.converged
    assert plan.max_defect < 1e-3
    assert plan.min_clearance > 0.0
    assert plan.t_end <= 60.0


def test_boundary_conditions(planned):
    planner, x0, goal, tols, plan = planned
    np.testing.assert_allclose(plan.X[:, 0], x0.as_ndarray(), atol=1e-6)
    np.testing.assert_allclose(plan.U[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(plan.U[:, -1], 0.0, atol=1e-6)
    xf = plan.X[:, -1]
    assert np.hypot(xf[0] - goal.target.x, xf[1] - goal.target.y) < goal.pos_tol
    assert abs(np.angle(np.exp(1j * (xf[2] - goal.target.psi)))) < goal.dir_tol
    assert np.hypot(xf[3] - goal.target.vx, xf[4] - goal.target.vy) < goal.vel_tol


def test_actuation_and_state_bounds(planned):
    planner, x0, goal, tols, plan = planned
    sp = planner.sp
    tol = 1e-6
    assert np.all(plan.U[0] >= sp.thrust_limits[0] - tol) and np.all(plan.U[0] <= sp.thrust_limits[1] + tol)
    assert np.all(plan.U[1] >= sp.ddelta_limits[0] - tol) and np.all(plan.U[1] <= sp.ddelta_limits[1] + tol)
    assert np.all(plan.dense_x[:, 6] >= sp.delta_limits[0] - tol) and np.all(plan.dense_x[:, 6] <= sp.delta_limits[1] + tol)
    assert np.all(np.abs(plan.X[3:5]) <= sp.vx_limits[1] + 1e-3)
    assert np.all(plan.dense_x[:, 7] >= sp.m_v)


def test_goal_reached_along_dense_plan(planned):
    """The goal check of the simulator succeeds on the planned (dense, nonlinear) trajectory."""
    from dg_commons.sim.models.spaceship import SpaceshipState

    planner, x0, goal, tols, plan = planned
    assert any(goal.is_fulfilled(SpaceshipState(*x)) for x in plan.dense_x[-20:])
    assert not goal.is_fulfilled(SpaceshipState(*plan.dense_x[0]))


def test_open_loop_zoh_close_to_plan(planned):
    """Applying the plan's inputs as 0.1 s zero-order hold (like the simulator) stays near the plan for a while."""
    planner, x0, goal, tols, plan = planned
    ts = np.arange(plan.t0, plan.t0 + min(plan.tf, 4.0), 0.1)
    xs = planner.integrator.simulate_zoh(plan.X[:, 0], lambda t: plan.u_at(t + 0.05), ts)
    ref = np.stack([plan.x_at(t) for t in ts])
    assert np.max(np.linalg.norm(xs[:, :2] - ref[:, :2], axis=1)) < 0.3


def test_sequences_for_report(planned):
    planner, x0, goal, tols, plan = planned
    cmds, states = plan.to_sequences()
    assert float(states.get_end()) >= 60.0
    s = states.at_interp(plan.t0 + 0.5 * plan.tf)
    assert np.isfinite(s.as_ndarray()).all()


def test_replan_from_perturbed_state(planned):
    planner, x0, goal, tols, plan = planned
    t = 0.3 * plan.tf
    x = plan.x_at(t).copy()
    x[0:2] += [0.3, -0.25]
    x[3:5] += [0.2, 0.15]
    x[2] += 0.1
    new = planner.replan(x, t, goal.target, tols, prev=plan, time_budget=10.0)
    assert new.max_defect < 0.02
    assert new.min_clearance > 0.0
    assert new.t0 == pytest.approx(t)
    np.testing.assert_allclose(new.X[:2, 0], x[:2], atol=1e-6)


def test_replan_with_disturbance_model(planned):
    planner, x0, goal, tols, plan = planned
    t = 0.25 * plan.tf
    x = plan.x_at(t)
    d = np.array([0.15, -0.1, 0.02])
    new = planner.replan(x, t, goal.target, tols, prev=plan, time_budget=10.0, disturbance=(d, 1.5))
    assert new.disturbance is not None
    np.testing.assert_allclose(new.disturbance_at(t + 0.1), d)
    np.testing.assert_allclose(new.disturbance_at(t + 2.0), 0.0)
    assert new.max_defect < 0.02 and new.min_clearance > 0.0


def test_initial_guess_is_collision_free_geometrically(planned):
    planner, x0, goal, tols, plan = planned
    path = planner._guess_path(x0.as_ndarray()[:2], np.array([goal.target.x, goal.target.y]), goal.target.psi)
    assert np.allclose(path[0], [x0.x, x0.y]) and np.allclose(path[-1], [goal.target.x, goal.target.y])
    body = path if planner.dock is None else path[:-1]  # last leg enters the dock
    for a, b in zip(body[:-1], body[1:]):
        assert planner._segment_free(a, b, 0.5)
