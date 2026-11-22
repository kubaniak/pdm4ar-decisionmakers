"""
End-to-end closed-loop simulations (slow). Run with e.g.

    python -m pytest tests/ex13 -m slow -n 4
"""
import numpy as np
import pytest

from scenario_utils import LOCAL_CONFIGS, PLAYER, PUBLIC, load, run

# reference: TA solution on the private cases ~870; ours on the public cases is ~880-910
PUBLIC_MIN_SCORE = {1: 890.0, 2: 850.0, 3: 880.0, 4: 860.0}


@pytest.mark.slow
@pytest.mark.parametrize("i", [1, 2, 3, 4], ids=lambda i: f"public{i}")
def test_public_case(i):
    ctx, avg, score = run(PUBLIC[i - 1], verbose=False)
    assert avg.collision_rate == 0
    assert avg.goal_success_rate == 1
    assert score > PUBLIC_MIN_SCORE[i]
    assert avg.avg_computation_time < 0.05


@pytest.mark.slow
@pytest.mark.parametrize("cfg", sorted(LOCAL_CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_local_case(cfg):
    ctx, avg, score = run(cfg, verbose=False)
    assert avg.collision_rate == 0, "collision"
    assert avg.goal_success_rate == 1, "goal not reached"
    assert score > 800


def test_agent_interface():
    """The agent keeps the simulator interface and exposes the plan for the report."""
    from dg_commons.sim import InitSimObservations, SimObservations, PlayerObservations
    from dg_commons import fd
    from decimal import Decimal
    from pdm4ar.exercises.ex13.agent import SpaceshipAgent

    ctx = load(PUBLIC[0])
    agent = ctx.players[PLAYER]
    assert type(agent) is SpaceshipAgent
    model = ctx.models[PLAYER]
    agent.on_episode_init(
        InitSimObservations(
            my_name=PLAYER, seed=0, model_geometry=model.model_geometry, model_params=model.model_params,
            dg_scenario=ctx.dg_scenario, goal=ctx.missions[PLAYER],
        )
    )
    obs = SimObservations(
        players=fd({PLAYER: PlayerObservations(state=model.get_state(), occupancy=None)}), time=Decimal("0")
    )
    cmd = agent.get_commands(obs)
    assert np.isfinite([cmd.thrust, cmd.ddelta]).all()
    assert agent.state_traj.at_interp(Decimal("59.9")) is not None
