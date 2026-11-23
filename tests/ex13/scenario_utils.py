"""Helpers to build planners / run simulations for ex13 scenarios outside of the report pipeline."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from dg_commons.sim.simulator import SimContext, Simulator

from pdm4ar.exercises.ex13.planner import SpaceshipPlanner, SolverParameters
from pdm4ar.exercises_def.ex13.goal import DockingTarget
from pdm4ar.exercises_def.ex13.perf_metrics import ex13_metrics
from pdm4ar.exercises_def.ex13.utils_config import sim_context_from_yaml

EX13_DEF = Path(__file__).resolve().parents[2] / "src" / "pdm4ar" / "exercises_def" / "ex13"
LOCAL_CONFIGS = Path(__file__).resolve().parent / "configs"
PUBLIC = [EX13_DEF / f"config_{i}_public.yaml" for i in range(1, 5)]

PLAYER = "PDM4ARSpaceship"


def load(path) -> SimContext:
    return sim_context_from_yaml(str(path))


def planner_from_context(ctx: SimContext, params: SolverParameters | None = None):
    agent = ctx.players[PLAYER]
    goal = ctx.missions[PLAYER]
    model = ctx.models[PLAYER]
    corners = None
    for so in ctx.dg_scenario.static_obstacles:
        if isinstance(so.shape, LineString):
            corners = np.array(so.shape.coords)
    planner = SpaceshipPlanner(
        planets=agent.planets,
        asteroids=agent.asteroids,
        sg=model.model_geometry,
        sp=model.model_params,
        boundary_corners=corners,
        dock=goal if isinstance(goal, DockingTarget) else None,
        params=params,
    )
    return planner, agent.init_state, goal


def run_simulation(ctx: SimContext) -> SimContext:
    """Like dg_commons.sim.utils.run_simulation, but with fresh per-instance simulator state.

    dg_commons keeps ``disabled_players`` (and other containers) as *class* attributes of ``Simulator``,
    so a collision in one run would disable the agent in every later run of the same process.
    """
    sim = Simulator()
    sim.disabled_players = []
    sim.last_commands = {}
    sim.simlogger = {}
    sim.run(ctx)
    return ctx


def run(path, verbose=True):
    ctx = load(path)
    t = time.perf_counter()
    run_simulation(ctx)
    wall = time.perf_counter() - t
    avg, per = ex13_metrics(ctx)
    score = avg.reduce_to_score()
    agent = ctx.log  # noqa
    if verbose:
        print(f"{Path(path).name}: score={score:.1f} goal={avg.goal_success_rate} coll={avg.collision_rate} "
              f"T={avg.avg_episode_duration:.2f} dist={avg.avg_distance_travelled:.2f} "
              f"eff={avg.avg_actuation_effort:.2f} d2g={avg.avg_distance2goal:.2f} "
              f"comp={avg.avg_computation_time:.4f} wall={wall:.1f}s")
    return ctx, avg, score
