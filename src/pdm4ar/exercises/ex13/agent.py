from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import LineString

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters

from pdm4ar.exercises.ex13.geometry import wrap_angle
from pdm4ar.exercises.ex13.planner import Plan, SpaceshipPlanner
from pdm4ar.exercises.ex13.tracker import TVLQRTracker
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
from pdm4ar.exercises_def.ex13.utils_plot import plot_traj


# HINT: as a good practice we suggest to use the config class to centralise activation of the debugging options
class Config:
    PLOT = False  # Optional planned/actual trajectory debugging (adds runtime).
    VERBOSE = False


@dataclass(frozen=True)
class MyAgentParams:
    """
    Replanning policy: the TV-LQR tracker absorbs small deviations; when the tracking error exceeds one
    of these thresholds (e.g. during an external disturbance), a new SCvx plan is computed from the
    measured state, warm-started from the current plan.
    """

    replan_pos_err: float = 0.35  # [m]
    replan_psi_err: float = 0.3  # [rad]
    replan_vel_err: float = 0.35  # [m/s]
    min_replan_interval: float = 0.8  # [s] between two replanning attempts
    plan_end_grace: float = 0.3  # [s] after the end of the plan before replanning to the goal
    # predicted-safety check: closed-loop rollout of the tracker (with the estimated disturbance assumed to
    # persist for its estimated remaining duration) and exact clearance of the predicted trajectory
    safety_check_interval: float = 0.3  # [s]
    safety_horizon: float = 6.0  # [s]
    safety_min_clearance: float = 0.05  # [m]
    safety_err_trigger: float = 0.08  # [m] only check when the tracking error or the disturbance is noticeable
    # an observed disturbance burst is assumed to last nominal_burst in total (public example: 2.5 s), and at
    # least / at most these remaining durations when planning and predicting
    nominal_burst: float = 2.5
    min_remaining_burst: float = 0.5
    max_remaining_burst: float = 2.0


class SpaceshipAgent(Agent):
    # How does it enter in the simulation? The SpaceshipAgent object is created as value
    # corresponding to key "PDM4ARSpaceship" in dict "players", which is an attribute of
    # SimContext returned by "sim_context_from_yaml" in utils_config.py
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SpaceshipState
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SpaceshipCommands]
    state_traj: DgSampledSequence[SpaceshipState]
    myname: PlayerName
    planner: SpaceshipPlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    def __init__(
        self,
        init_state: SpaceshipState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SpaceshipAgent with information about its environment, i.e. planet and spaceship parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.params = MyAgentParams()
        self.planets = planets
        self.asteroids = asteroids

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.

        the time spent in this method is **not** considered in the score.
        """
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        self.goal = init_sim_obs.goal
        self.goal_state = init_sim_obs.goal.target
        self.goal_tols = (self.goal.pos_tol, self.goal.vel_tol, self.goal.dir_tol)

        # Map boundary (LineString among the static obstacles); planets come from the constructor and
        # the dock geometry from the goal (DockingTarget is a subclass of SpaceshipTarget).
        boundary = None
        if init_sim_obs.dg_scenario is not None:
            for so in init_sim_obs.dg_scenario.static_obstacles:
                if isinstance(so.shape, LineString):
                    boundary = np.array(so.shape.coords)
        dock = self.goal if isinstance(self.goal, DockingTarget) else None

        self.planner = SpaceshipPlanner(
            planets=self.planets,
            asteroids=self.asteroids,
            sg=self.sg,
            sp=self.sp,
            boundary_corners=boundary,
            dock=dock,
        )
        self.tracker = TVLQRTracker(
            integrator=self.planner.integrator,
            thrust_limits=self.sp.thrust_limits,
            ddelta_limits=self.sp.ddelta_limits,
            delta_limits=self.sp.delta_limits,
        )

        plan = self.planner.plan_initial(self.init_state, self.goal_state, self.goal_tols, t0=0.0)
        self.plans: list[Plan] = []
        self.last_replan_t = -np.inf
        self.last_safety_check_t = -np.inf
        self.dist_start = None
        self.n_replans = 0
        self._set_plan(plan)
        if Config.VERBOSE:
            print(
                f"[agent] initial plan {plan.label}: tf={plan.tf:.2f}s feasible={plan.feasible} "
                f"converged={plan.converged} it={plan.iterations} clearance={plan.min_clearance:.3f}"
            )

    def _set_plan(self, plan: Plan):
        self.plan = plan
        self.plans.append(plan)
        self.tracker.set_plan(plan)
        self.cmds_plan, self.state_traj = plan.to_sequences()

    def get_commands(self, sim_obs: SimObservations) -> SpaceshipCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)

        NOTE: this function is not run in real time meaning that the simulation is stopped when the function is called.
        Thus the time efficiency of the replanning is not critical for the simulation.
        However the average time spent in get_commands is still considered in the score.

        Do **not** modify the signature of this method.
        """
        current_state = sim_obs.players[self.myname].state
        self.actual_trajectory.append(current_state)
        t = float(sim_obs.time)
        x = current_state.as_ndarray().astype(float)

        # plotting the trajectory every 2.5 sec (this is optional, for better visualization)
        if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)

        self.tracker.observe(t, x)  # disturbance estimate from the last step's prediction residual
        if np.any(self.tracker.d_hat != 0.0):
            if self.dist_start is None:
                self.dist_start = t - self.tracker.params.dt
        else:
            self.dist_start = None

        if self._needs_replan(t, x):
            self._replan(t, x)

        u = self.tracker.command(t, x)
        self.tracker.record_command(t, x, u)
        return SpaceshipCommands(thrust=float(u[0]), ddelta=float(u[1]))

    def _needs_replan(self, t: float, x: np.ndarray) -> bool:
        P = self.params
        if t - self.last_replan_t < P.min_replan_interval:
            return False
        if t > self.plan.t_end + P.plan_end_grace:
            return True  # plan finished but the goal was not reached
        e = self.tracker.error(t, x)
        if (
            np.linalg.norm(e[0:2]) > P.replan_pos_err
            or abs(e[2]) > P.replan_psi_err
            or np.linalg.norm(e[3:5]) > P.replan_vel_err
        ):
            return True
        disturbed = np.any(self.tracker.d_hat != 0.0) or np.linalg.norm(e[0:2]) > P.safety_err_trigger
        if disturbed and t - self.last_safety_check_t >= P.safety_check_interval:
            self.last_safety_check_t = t
            return self._predicted_clearance(t, x) < P.safety_min_clearance
        return False

    def _predicted_clearance(self, t: float, x: np.ndarray) -> float:
        """Exact clearance of the predicted closed-loop trajectory under the current plan."""
        P = self.params
        t_end = min(self.plan.t_end, t + P.safety_horizon)
        if t_end - t < 0.2:
            return np.inf
        x_c = self._continuous_heading(t, x)
        ts, xs, _ = self.tracker.rollout(x_c, t, t_end, d_hat=self.tracker.d_hat, d_duration=self._remaining_burst(t))
        return float(np.min(self.planner.checker.clearance(xs, ts)))

    def _remaining_burst(self, t: float) -> float:
        P = self.params
        if self.dist_start is None:
            return 0.0
        return float(np.clip(P.nominal_burst - (t - self.dist_start), P.min_remaining_burst, P.max_remaining_burst))

    def _continuous_heading(self, t: float, x: np.ndarray) -> np.ndarray:
        x_c = x.copy()
        psi_ref = self.plan.x_at(t)[2]
        x_c[2] = psi_ref + wrap_angle(x[2] - psi_ref)
        return x_c

    def _replan(self, t: float, x: np.ndarray):
        self.last_replan_t = t
        P = self.params
        rollout = None
        current_clearance = np.inf
        plan_over = t > self.plan.t_end
        if self.plan.t_end - t > 1.0:
            x_c = self._continuous_heading(t, x)
            rollout = self.tracker.rollout(
                x_c, t, self.plan.t_end, d_hat=self.tracker.d_hat, d_duration=self._remaining_burst(t)
            )
            ts, xs, _ = self.tracker.rollout(
                x_c, t, min(self.plan.t_end, t + P.safety_horizon), d_hat=self.tracker.d_hat,
                d_duration=self._remaining_burst(t),
            )
            current_clearance = float(np.min(self.planner.checker.clearance(xs, ts)))
        urgent = current_clearance <= 0.0
        disturbance = None
        if np.any(self.tracker.d_hat != 0.0):
            disturbance = (self.tracker.d_hat.copy(), self._remaining_burst(t))
        new = self.planner.replan(
            x, t, self.goal_state, self.goal_tols, prev=self.plan, rollout=rollout, urgent=urgent,
            disturbance=disturbance,
        )
        consistent = new is not None and new.max_defect < 0.02 and new.min_clearance > 0.0
        # take a feasible plan; otherwise a dynamically consistent, collision-free one if it is safer than going on
        accept = new is not None and (
            new.feasible
            or (consistent and (plan_over or new.min_clearance > min(current_clearance, P.safety_min_clearance)))
        )
        if Config.VERBOSE:
            print(
                f"[agent] t={t:.1f} replan {new.label}: tf={new.tf:.2f}s feasible={new.feasible} "
                f"it={new.iterations} {new.solve_time:.2f}s clearance={new.min_clearance:.3f} defect={new.max_defect:.3g} "
                f"viol={new.max_violation:.3g} conv={new.converged} accepted={accept}"
            )
        if accept:
            self.n_replans += 1
            self._set_plan(new)
