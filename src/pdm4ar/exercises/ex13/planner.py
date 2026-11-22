from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Optional, Union

import cvxpy as cvx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters

from pdm4ar.exercises.ex13.discretization import BatchFirstOrderHold, FirstOrderHold
from pdm4ar.exercises.ex13.geometry import (
    ExactCollisionChecker,
    ObstacleSet,
    ShipCircles,
    boundary_halfplanes,
    dock_segments,
    ship_circles,
    static_geometries,
    wrap_angle,
)
from pdm4ar.exercises.ex13.spaceship import SpaceshipDyn
from pdm4ar.exercises_def.ex13.utils_params import AsteroidParams, PlanetParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 40  # max SCvx iterations for the initial plan
    replan_max_iterations: int = 12  # max SCvx iterations when replanning online
    replan_time_budget: float = 2.0  # [s] wall-clock budget of an online replan
    urgent_time_budget: float = 5.0  # [s] budget when the current plan is predicted to collide
    urgent_time_scales: tuple = (1.0, 1.4, 0.75)  # duration scalings of fresh guesses when urgent
    init_extra_time_scales: tuple = (1.4, 0.75, 1.8)  # extra initial candidates if none is feasible

    # SCvx penalty weights
    lambda_nu: float = 1e5  # virtual control (dynamics defect) weight, >> obstacle slack weight
    lambda_s: float = 1e4  # obstacle slack weight
    lambda_v: float = 1e3  # velocity-box slack weight (soft: preferred over dynamics defects)

    # Objective weights, mirroring the score: duration/3 + distance/5 + mean|thrust|  (x10)
    w_time: float = 10.0 / 3.0
    w_effort: float = 10.0
    w_dist: float = 2.0
    w_ddelta: float = 0.02  # smoothness regularisation of the nozzle rate
    w_dthrust: float = 0.05  # smoothness regularisation of the thrust

    # Trust region
    tr_radius: float = 5.0  # initial trust region radius
    min_tr_radius: float = 1e-3  # min trust region radius
    max_tr_radius: float = 50.0  # max trust region radius
    rho_0: float = 0.0
    rho_1: float = 0.25
    rho_2: float = 0.9
    alpha: float = 2.0  # div factor trust region update
    beta: float = 2.0  # mult factor trust region update
    tr_scale_x: tuple = (1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1)
    tr_scale_u: tuple = (1.0, 0.5)
    tr_scale_p: float = 2.0

    check_dynamics: bool = True  # compare the SymPy model with the simulator model at start-up

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # RK4 sub-steps per interval
    stop_crit: float = 1e-3  # relative predicted-decrease stopping criterion

    # Margins keeping control authority for the feedback tracker
    thrust_margin: float = 0.85
    ddelta_margin: float = 0.9
    delta_margin: float = 0.85
    vel_margin: float = 0.95

    # Obstacle inflation [m] (on top of the circle cover of the spaceship)
    planet_margin: float = 0.12
    asteroid_margin: float = 0.3
    dock_margin: float = 0.06
    boundary_margin: float = 0.1

    # Terminal set as a fraction of the goal tolerances
    term_pos_frac: float = 0.3
    term_vel_frac: float = 0.3
    term_dir_frac: float = 0.3

    p_min: float = 0.5  # minimum trajectory duration
    t_max: float = 59.0  # trajectories must end before this absolute time
    max_defect: float = 2e-2  # dynamics defect accepted for a feasible plan
    max_violation: float = 2e-2  # obstacle violation accepted for a feasible plan [m]

    # Initial guess
    grid_res: float = 0.25
    guess_clearance: float = 1.0
    predock_dist: float = 3.0


@dataclass
class Plan:
    t0: float
    tf: float  # duration
    X: NDArray  # (8, K)
    U: NDArray  # (2, K)
    dense_t: NDArray
    dense_x: NDArray  # (Nd, 8)
    cost: float = np.inf
    converged: bool = False
    feasible: bool = False
    iterations: int = 0
    max_defect: float = np.inf
    max_violation: float = np.inf
    min_clearance: float = -np.inf
    solve_time: float = 0.0
    label: str = ""
    disturbance: Optional[tuple] = None  # (d, duration) assumed by the plan from t0

    def disturbance_at(self, t: float) -> NDArray:
        if self.disturbance is None or not (self.t0 <= t < self.t0 + self.disturbance[1]):
            return np.zeros(3)
        return np.asarray(self.disturbance[0], dtype=float)

    @property
    def t_end(self) -> float:
        return self.t0 + self.tf

    @property
    def node_times(self) -> NDArray:
        return self.t0 + self.tf * np.linspace(0.0, 1.0, self.X.shape[1])

    def u_at(self, t: float) -> NDArray:
        tn = self.node_times
        t = float(np.clip(t, tn[0], tn[-1]))
        return np.array([np.interp(t, tn, self.U[i]) for i in range(self.U.shape[0])])

    def x_at(self, t: float) -> NDArray:
        t = float(np.clip(t, self.dense_t[0], self.dense_t[-1]))
        return np.array([np.interp(t, self.dense_t, self.dense_x[:, i]) for i in range(self.dense_x.shape[1])])

    def to_sequences(self, horizon: float = 61.0):
        tn = self.node_times
        commands = [SpaceshipCommands(*u) for u in self.U.T]
        cmd_t = list(tn)
        states = [SpaceshipState(*x) for x in self.dense_x]
        st_t = list(self.dense_t)
        if st_t[-1] < horizon:
            st_t.append(horizon)
            states.append(states[-1])
            cmd_t.append(horizon)
            commands.append(SpaceshipCommands(0.0, 0.0))
        return (
            DgSampledSequence[SpaceshipCommands](timestamps=cmd_t, values=commands),
            DgSampledSequence[SpaceshipState](timestamps=st_t, values=states),
        )


class SpaceshipPlanner:
    """
    SCvx trajectory planner (Mao et al. 2018 / Malyuta et al. 2021) for the spaceship with
    free final time, static planets, linearly moving asteroids, dock structure and map boundary.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
        boundary_corners: Optional[NDArray] = None,
        dock=None,
        params: Optional[SolverParameters] = None,
    ):
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp
        self.params = params or SolverParameters()
        self.boundary_corners = None if boundary_corners is None else np.asarray(boundary_corners, dtype=float)
        self.dock = dock
        self.debug = False
        self.disturbance = None  # (d, duration) assumed to act from the start of the plan being optimised

        # Spaceship dynamics (SymPy model, checked against the simulator model)
        self.spaceship = SpaceshipDyn(self.sg, self.sp)
        if self.params.check_dynamics:
            reference = FirstOrderHold(self.spaceship, 3, 2)
            if not reference.check_dynamics():
                raise ValueError("Dynamics check failed: SpaceshipDyn.get_dynamics() differs from the simulator.")
        self.n_x, self.n_u, self.n_p = self.spaceship.n_x, self.spaceship.n_u, self.spaceship.n_p

        # Fast vectorised discretization
        self.K = self.params.K
        self.tau = np.linspace(0.0, 1.0, self.K)
        self.integrator = BatchFirstOrderHold(sg.l_r, sg.Iz, sp.C_T, self.K, self.params.N_sub)

        # Geometry
        self.circles: ShipCircles = ship_circles(sg)
        self.obstacles = self._build_obstacles()
        self.M = self.obstacles.n_obs * self.circles.n
        self.checker = self._build_checker()
        self._build_grid()

        # Bounds
        P = self.params
        self.thrust_lim = (sp.thrust_limits[0] * P.thrust_margin, sp.thrust_limits[1] * P.thrust_margin)
        self.ddelta_lim = (sp.ddelta_limits[0] * P.ddelta_margin, sp.ddelta_limits[1] * P.ddelta_margin)
        self.delta_lim = (sp.delta_limits[0] * P.delta_margin, sp.delta_limits[1] * P.delta_margin)
        self.vel_lim = (sp.vx_limits[0] * P.vel_margin, sp.vx_limits[1] * P.vel_margin)

        # Cvx problem (DPP: compiled once, re-solved with new parameter values)
        self.variables = self._get_variables()
        self.problem_parameters = self._get_problem_parameters()
        constraints = self._get_constraints()
        objective = self._get_objective()
        self.problem = cvx.Problem(objective, constraints)

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

    # ================================================================================================ setup
    def _build_obstacles(self) -> ObstacleSet:
        P = self.params
        obs = ObstacleSet()
        c0, v, r = [], [], []
        for pl in self.planets.values():
            c0.append(pl.center)
            v.append([0.0, 0.0])
            r.append(pl.radius + P.planet_margin)
        for a in self.asteroids.values():
            co, so = np.cos(a.orientation), np.sin(a.orientation)
            vw = np.array([co * a.velocity[0] - so * a.velocity[1], so * a.velocity[0] + co * a.velocity[1]])
            c0.append(a.start)
            v.append(vw)
            r.append(a.radius + P.asteroid_margin)
        if c0:
            obs.circ_c0 = np.array(c0, dtype=float)
            obs.circ_v = np.array(v, dtype=float)
            obs.circ_r = np.array(r, dtype=float)
        if self.dock is not None:
            a, b = dock_segments(self.dock)
            obs.seg_a, obs.seg_b = a, b
            obs.seg_r = np.full(len(a), 0.05 + P.dock_margin)
        if self.boundary_corners is not None:
            n, w = boundary_halfplanes(self.boundary_corners)
            obs.hp_n, obs.hp_w = n, w
            obs.hp_r = np.full(len(w), P.boundary_margin)
        self.n_planets = len(self.planets)
        return obs

    def _build_checker(self) -> ExactCollisionChecker:
        dock_geom = self.dock.get_plottable_geometry() if self.dock is not None else None
        geoms = static_geometries(self.planets, self.boundary_corners, dock_geom)
        n_p = self.n_planets
        return ExactCollisionChecker(
            sg=self.sg,
            static_geoms=geoms,
            asteroid_c0=self.obstacles.circ_c0[n_p:],
            asteroid_v=self.obstacles.circ_v[n_p:],
            asteroid_r=np.array([a.radius for a in self.asteroids.values()]),
        )

    def _get_variables(self) -> dict:
        K, M = self.K, self.M
        variables = {
            "X": cvx.Variable((self.n_x, K)),
            "U": cvx.Variable((self.n_u, K)),
            "p": cvx.Variable(self.n_p),
            "nu": cvx.Variable((self.n_x, K - 1)),
            "s_v": cvx.Variable(nonneg=True),  # soft velocity-box slack (exact penalty)
        }
        if M > 0:
            variables["s"] = cvx.Variable((1, K), nonneg=True)  # one obstacle slack per node
        return variables

    def _get_problem_parameters(self) -> dict:
        K, M, nx, nu = self.K, self.M, self.n_x, self.n_u
        pp = {
            "init_state": cvx.Parameter(nx),
            "u0_lb": cvx.Parameter(nu),
            "u0_ub": cvx.Parameter(nu),
            "p_lb": cvx.Parameter(nonneg=True),
            "p_ub": cvx.Parameter(nonneg=True),
            "v_lb": cvx.Parameter(),
            "v_ub": cvx.Parameter(),
            "d_lb": cvx.Parameter(),
            "d_ub": cvx.Parameter(),
            "goal_xy": cvx.Parameter(2),
            "goal_psi": cvx.Parameter(),
            "goal_v": cvx.Parameter(2),
            "r_pos": cvx.Parameter(nonneg=True),
            "r_psi": cvx.Parameter(nonneg=True),
            "r_vel": cvx.Parameter(nonneg=True),
            "A": [cvx.Parameter((nx, K - 1)) for _ in range(nx)],
            "Bm": [cvx.Parameter((nx, K - 1)) for _ in range(nu)],
            "Bp": [cvx.Parameter((nx, K - 1)) for _ in range(nu)],
            "F": cvx.Parameter((nx, K - 1)),
            "r": cvx.Parameter((nx, K - 1)),
            "X_ref": cvx.Parameter((nx, K)),
            "U_ref": cvx.Parameter((nu, K)),
            "p_ref": cvx.Parameter(nonneg=True),
            "eta": cvx.Parameter(nonneg=True),
        }
        if M > 0:
            for name in ("Gx", "Gy", "Gpsi", "Gp", "h"):
                pp[name] = cvx.Parameter((M, K))
        return pp

    def _get_constraints(self) -> list[cvx.Constraint]:
        X, U, p, nu = self.variables["X"], self.variables["U"], self.variables["p"], self.variables["nu"]
        pp = self.problem_parameters
        K, M, nx = self.K, self.M, self.n_x
        ones_x = np.ones((nx, 1))
        P = self.params

        # discretized linear dynamics x_{k+1} = A x_k + B- u_k + B+ u_{k+1} + F p + r + nu
        rhs = pp["r"] + nu + cvx.multiply(pp["F"], p[0])
        for j in range(nx):
            rhs = rhs + cvx.multiply(pp["A"][j], ones_x @ X[j : j + 1, :-1])
        for j in range(self.n_u):
            rhs = rhs + cvx.multiply(pp["Bm"][j], ones_x @ U[j : j + 1, :-1])
            rhs = rhs + cvx.multiply(pp["Bp"][j], ones_x @ U[j : j + 1, 1:])

        constraints = [
            X[:, 0] == pp["init_state"],
            X[:, 1:] == rhs,
            U[:, 0] >= pp["u0_lb"],
            U[:, 0] <= pp["u0_ub"],
            U[:, K - 1] == 0,
            U[0, :] >= self.thrust_lim[0],
            U[0, :] <= self.thrust_lim[1],
            U[1, :] >= self.ddelta_lim[0],
            U[1, :] <= self.ddelta_lim[1],
            X[6, 1:] >= pp["d_lb"],
            X[6, 1:] <= pp["d_ub"],
            X[3:5, 1:] >= pp["v_lb"] - self.variables["s_v"],
            X[3:5, 1:] <= pp["v_ub"] + self.variables["s_v"],
            X[7, :] >= self.sp.m_v,
            p >= pp["p_lb"],
            p <= pp["p_ub"],
            # terminal set inside the goal tolerances
            cvx.norm(X[0:2, K - 1] - pp["goal_xy"], 2) <= pp["r_pos"],
            cvx.abs(X[2, K - 1] - pp["goal_psi"]) <= pp["r_psi"],
            cvx.norm(X[3:5, K - 1] - pp["goal_v"], 2) <= pp["r_vel"],
            X[5, K - 1] == 0,
        ]

        # linearised obstacle avoidance  h + Gx x + Gy y + Gpsi psi + Gp p + s >= 0
        if M > 0:
            ones_m = np.ones((M, 1))
            g_lin = (
                pp["h"]
                + cvx.multiply(pp["Gx"], ones_m @ X[0:1, :])
                + cvx.multiply(pp["Gy"], ones_m @ X[1:2, :])
                + cvx.multiply(pp["Gpsi"], ones_m @ X[2:3, :])
                + cvx.multiply(pp["Gp"], p[0])
            )
            constraints.append(g_lin + ones_m @ self.variables["s"] >= 0)

        # trust region (scaled, per node)
        Wx = np.tile((1.0 / np.array(P.tr_scale_x))[:, None], (1, K))
        Wu = np.tile((1.0 / np.array(P.tr_scale_u))[:, None], (1, K))
        tr = (
            cvx.norm(cvx.multiply(Wx, X - pp["X_ref"]), 2, axis=0)
            + cvx.norm(cvx.multiply(Wu, U - pp["U_ref"]), 2, axis=0)
            + cvx.abs(p[0] - pp["p_ref"]) / P.tr_scale_p
        )
        constraints.append(tr <= pp["eta"])
        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        X, U, p, nu = self.variables["X"], self.variables["U"], self.variables["p"], self.variables["nu"]
        P = self.params
        K = self.K
        obj = (
            P.w_time * p[0]
            + P.w_effort * cvx.sum(cvx.abs(U[0, :])) / K
            + P.w_dist * cvx.sum(cvx.norm(X[0:2, 1:] - X[0:2, :-1], 2, axis=0))
            + P.w_ddelta * cvx.sum_squares(U[1, :])
            + P.w_dthrust * cvx.sum_squares(U[0, 1:] - U[0, :-1])
            + P.lambda_nu * cvx.sum(cvx.abs(nu))
            + P.lambda_v * self.variables["s_v"]
        )
        if self.M > 0:
            obj = obj + P.lambda_s * cvx.sum(self.variables["s"])
        return cvx.Minimize(obj)

    # ================================================================================================ costs
    def _objective_np(self, X, U, p) -> float:
        P = self.params
        return float(
            P.w_time * p[0]
            + P.w_effort * np.sum(np.abs(U[0])) / self.K
            + P.w_dist * np.sum(np.linalg.norm(np.diff(X[0:2], axis=1), axis=0))
            + P.w_ddelta * np.sum(U[1] ** 2)
            + P.w_dthrust * np.sum(np.diff(U[0]) ** 2)
        )

    def _evaluate(self, X, U, p, t0) -> dict:
        """Discretize at (X, U, p) and compute the nonlinear SCvx cost."""
        disc = self.integrator.calculate_discretization(X, U, p, disturbance=self.disturbance)
        defect = disc["x_next"].T - X[:, 1:]
        g, grad, gp = self.obstacles.evaluate(X, float(p[0]), t0, self.tau, self.circles)
        viol = np.maximum(0.0, -g)
        P = self.params
        node_viol = np.max(viol, axis=0) if viol.size else np.zeros(1)
        pp = self.problem_parameters
        v_viol = max(0.0, float(np.max(np.abs(X[3:5, 1:]))) - float(pp["v_ub"].value))
        J = (
            self._objective_np(X, U, p)
            + P.lambda_nu * np.sum(np.abs(defect))
            + P.lambda_s * np.sum(node_viol)
            + P.lambda_v * v_viol
        )
        return {
            "disc": disc,
            "J": J,
            "defect": defect,
            "g": g,
            "grad": grad,
            "gp": gp,
            "max_defect": float(np.max(np.abs(defect))) if defect.size else 0.0,
            "max_violation": float(np.max(viol)) if viol.size else 0.0,
        }

    def _populate(self, X, U, p, ev, eta):
        pp = self.problem_parameters
        d = ev["disc"]
        for j in range(self.n_x):
            pp["A"][j].value = d["A"][:, :, j].T
        for j in range(self.n_u):
            pp["Bm"][j].value = d["Bm"][:, :, j].T
            pp["Bp"][j].value = d["Bp"][:, :, j].T
        pp["F"].value = d["F"].T
        pp["r"].value = d["r"].T
        if self.M > 0:
            g, grad, gp = ev["g"], ev["grad"], ev["gp"]
            pp["Gx"].value = grad[..., 0]
            pp["Gy"].value = grad[..., 1]
            pp["Gpsi"].value = grad[..., 2]
            pp["Gp"].value = gp
            pp["h"].value = g - grad[..., 0] * X[0] - grad[..., 1] * X[1] - grad[..., 2] * X[2] - gp * p[0]
        pp["X_ref"].value = X
        pp["U_ref"].value = U
        pp["p_ref"].value = float(p[0])
        pp["eta"].value = float(eta)

    def _set_boundary_conditions(self, x0, u0_bounds, goal_state, goal_tols, t0, v_init_abs):
        pp = self.problem_parameters
        P = self.params
        pp["init_state"].value = x0
        pp["u0_lb"].value = u0_bounds[0]
        pp["u0_ub"].value = u0_bounds[1]
        pp["p_lb"].value = P.p_min
        pp["p_ub"].value = max(P.t_max - t0, P.p_min + 0.1)
        vmax = max(self.vel_lim[1], v_init_abs + 0.05)
        pp["v_lb"].value = -vmax
        pp["v_ub"].value = vmax
        # the margin on the nozzle angle must not make the measured initial state infeasible
        pp["d_lb"].value = min(self.delta_lim[0], float(x0[6]) - 1e-3)
        pp["d_ub"].value = max(self.delta_lim[1], float(x0[6]) + 1e-3)
        pp["goal_xy"].value = np.array([goal_state.x, goal_state.y])
        pp["goal_psi"].value = float(goal_state.psi)
        pp["goal_v"].value = np.array([goal_state.vx, goal_state.vy])
        pos_tol, vel_tol, dir_tol = goal_tols
        pp["r_pos"].value = P.term_pos_frac * pos_tol
        pp["r_vel"].value = P.term_vel_frac * vel_tol
        pp["r_psi"].value = P.term_dir_frac * dir_tol

    # ================================================================================================ SCvx
    def scvx(
        self,
        X: NDArray,
        U: NDArray,
        p: NDArray,
        t0: float,
        max_iter: int,
        time_budget: float = np.inf,
        label: str = "",
    ) -> Plan:
        """Run SCvx from the reference (X, U, p). Boundary conditions must already be set."""
        P = self.params
        start = _time.perf_counter()
        X, U, p = X.copy(), U.copy(), np.array(p, dtype=float).reshape(1).copy()
        X[:, 0] = self.problem_parameters["init_state"].value
        ev = self._evaluate(X, U, p, t0)
        eta = P.tr_radius
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            self._populate(X, U, p, ev, eta)
            try:
                self.problem.solve(solver=P.solver, verbose=P.verbose_solver, ignore_dpp=False)
            except cvx.SolverError:
                eta /= P.alpha
                if eta < P.min_tr_radius:
                    break
                continue
            if self.problem.status not in ("optimal", "optimal_inaccurate"):
                eta /= P.alpha
                if eta < P.min_tr_radius:
                    break
                continue
            L_new = float(self.problem.value)
            X_new = self.variables["X"].value
            U_new = self.variables["U"].value
            p_new = np.maximum(self.variables["p"].value, P.p_min)
            ev_new = self._evaluate(X_new, U_new, p_new, t0)

            pred = ev["J"] - L_new
            act = ev["J"] - ev_new["J"]
            if pred <= P.stop_crit * max(1.0, abs(ev["J"])):
                if ev_new["J"] <= ev["J"] + 1e-6 * max(1.0, abs(ev["J"])):
                    X, U, p, ev = X_new, U_new, p_new, ev_new
                converged = True
                break
            rho = act / pred
            if self.debug:
                print(
                    f"    it {it:2d} J={ev['J']:.3f} L={L_new:.3f} Jn={ev_new['J']:.3f} rho={rho:.3f} eta={eta:.3g} "
                    f"def={ev_new['max_defect']:.2e} viol={ev_new['max_violation']:.2e} p={p_new[0]:.2f}"
                )
            if rho < P.rho_0:
                eta = max(eta / P.alpha, P.min_tr_radius)
            else:
                X, U, p, ev = X_new, U_new, p_new, ev_new
                if rho < P.rho_1:
                    eta = max(eta / P.alpha, P.min_tr_radius)
                elif rho >= P.rho_2:
                    eta = min(eta * P.beta, P.max_tr_radius)
            if eta <= P.min_tr_radius:
                break
            if _time.perf_counter() - start > time_budget:
                break

        return self._make_plan(X, U, p, t0, ev, converged, it, _time.perf_counter() - start, label)

    def _make_plan(self, X, U, p, t0, ev, converged, iterations, solve_time, label) -> Plan:
        P = self.params
        disc = self.integrator.calculate_discretization(X, U, p, dense=True, disturbance=self.disturbance)
        xd = disc["x_dense"]  # (K-1, N_sub+1, 8): piecewise propagation from each node
        n_sub = xd.shape[1] - 1
        dense_x = np.concatenate([xd[:, :-1, :].reshape(-1, 8), X[:, -1][None]], axis=0)
        dense_tau = np.concatenate(
            [(np.arange(self.K - 1)[:, None] + np.arange(n_sub)[None] / n_sub).reshape(-1), [self.K - 1]]
        ) / (self.K - 1)
        dense_t = t0 + float(p[0]) * dense_tau
        clearance = self.checker.clearance(dense_x, dense_t)
        min_clear = float(np.min(clearance))
        feasible = (
            ev["max_defect"] <= P.max_defect
            and ev["max_violation"] <= P.max_violation
            and min_clear > 0.0
            and t0 + float(p[0]) <= 60.0
        )
        return Plan(
            t0=t0,
            tf=float(p[0]),
            X=X,
            U=U,
            dense_t=dense_t,
            dense_x=dense_x,
            cost=self._objective_np(X, U, p),
            converged=converged,
            feasible=bool(feasible),
            iterations=iterations,
            max_defect=ev["max_defect"],
            max_violation=ev["max_violation"],
            min_clearance=min_clear,
            solve_time=solve_time,
            label=label,
            disturbance=self.disturbance,
        )

    # ================================================================================================ API
    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState, goal_tols=(0.5, 1.0, 0.5)
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """Compute a trajectory from init_state to goal_state (skeleton interface)."""
        plan = self.plan_initial(init_state, goal_state, goal_tols)
        self.last_plan = plan
        return plan.to_sequences()

    def plan_initial(self, init_state: SpaceshipState, goal_state: DynObstacleState, goal_tols, t0: float = 0.0):
        """Multi-start SCvx from the initial state; returns the best feasible plan (or the least bad)."""
        self.init_state = init_state
        self.goal_state = goal_state
        x0 = init_state.as_ndarray().astype(float)
        u0 = np.zeros(2)
        best: Optional[Plan] = None
        candidates = []
        for psi_goal in self._goal_heading_options(x0[2], goal_state.psi):
            candidates.append(psi_goal)
        runs = [(psi, 1.0) for psi in candidates]
        extra = [(psi, sc) for sc in self.params.init_extra_time_scales for psi in candidates]
        for k, (psi_goal, scale) in enumerate(runs + extra):
            if k >= len(runs) and best is not None and best.feasible:
                break
            g = _with_psi(goal_state, psi_goal)
            self._set_boundary_conditions(x0, (u0, u0), g, goal_tols, t0, np.max(np.abs(x0[3:5])))
            X, U, p = self.initial_guess(x0, g, t0, time_scale=scale)
            plan = self.scvx(X, U, p, t0, self.params.max_iterations, label=f"init-{k}-x{scale}")
            if best is None or _better(plan, best):
                best = plan
        return best

    def replan(
        self,
        x0: NDArray,
        t0: float,
        goal_state: DynObstacleState,
        goal_tols,
        prev: Optional[Plan] = None,
        time_budget: Optional[float] = None,
        rollout: Optional[tuple] = None,
        urgent: bool = False,
        disturbance: Optional[tuple] = None,
    ) -> Plan:
        """
        :param rollout: optional (times, states, commands) closed-loop prediction of the tracker on the
            nominal model from x0; it is dynamically consistent and is used as the first warm start.
        :param disturbance: optional (d, duration): estimated disturbance assumed to persist for `duration`.
        """
        self.disturbance = disturbance
        try:
            return self._replan(x0, t0, goal_state, goal_tols, prev, time_budget, rollout, urgent)
        finally:
            self.disturbance = None

    def _replan(self, x0, t0, goal_state, goal_tols, prev, time_budget, rollout, urgent) -> Plan:
        P = self.params
        budget = (P.urgent_time_budget if urgent else P.replan_time_budget) if time_budget is None else time_budget
        start = _time.perf_counter()
        x0 = np.asarray(x0, dtype=float)
        if prev is not None:
            # keep heading continuous with the plan
            x0 = x0.copy()
            x0[2] = prev.x_at(t0)[2] + wrap_angle(x0[2] - prev.x_at(t0)[2])
        psi_goal = x0[2] + wrap_angle(goal_state.psi - x0[2])
        if prev is not None:
            psi_goal = prev.X[2, -1] + wrap_angle(goal_state.psi - prev.X[2, -1])
        g = _with_psi(goal_state, psi_goal)
        u_bounds = (np.array([self.thrust_lim[0], self.ddelta_lim[0]]), np.array([self.thrust_lim[1], self.ddelta_lim[1]]))
        self._set_boundary_conditions(x0, u_bounds, g, goal_tols, t0, np.max(np.abs(x0[3:5])))

        plans = []
        if rollout is not None and prev is not None and prev.t_end - t0 > 1.0:
            X, U, p = self._rollout_start(rollout, x0, t0, g)
            plans.append(self.scvx(X, U, p, t0, P.replan_max_iterations, budget, label="rollout"))
            if plans[-1].feasible:
                return plans[-1]
        remaining = budget - (_time.perf_counter() - start)
        if prev is not None and prev.t_end - t0 > 1.0 and remaining > 0.3:
            X, U, p = self._warm_start(prev, x0, t0)
            plans.append(self.scvx(X, U, p, t0, P.replan_max_iterations, remaining, label="warm"))
            if plans[-1].feasible:
                return plans[-1]
        remaining = budget - (_time.perf_counter() - start)
        scales = P.urgent_time_scales if urgent else (1.0,)
        for scale in scales:
            remaining = budget - (_time.perf_counter() - start)
            if remaining < 0.2:
                break
            X, U, p = self.initial_guess(x0, g, t0, time_scale=scale)
            plans.append(self.scvx(X, U, p, t0, P.replan_max_iterations, remaining, label=f"fresh-x{scale}"))
            if plans[-1].feasible:
                break
        best = None
        for pl in plans:
            if best is None or _better(pl, best):
                best = pl
        return best

    def _rollout_start(self, rollout, x0: NDArray, t0: float, goal_state):
        ts, xs, us = rollout
        T = max(ts[-1] - t0, self.params.p_min + 0.5)
        tk = t0 + T * self.tau
        X = np.stack([np.interp(tk, ts, xs[:, i]) for i in range(8)])
        U = np.stack([np.interp(tk, ts, us[:, i]) for i in range(2)])
        X[:, 0] = x0
        # the terminal set is a hard constraint: end the reference exactly at the goal
        X[0, -1], X[1, -1], X[2, -1] = goal_state.x, goal_state.y, goal_state.psi
        X[3, -1], X[4, -1], X[5, -1] = goal_state.vx, goal_state.vy, 0.0
        U[:, -1] = 0.0
        U[0] = np.clip(U[0], *self.thrust_lim)
        U[1] = np.clip(U[1], *self.ddelta_lim)
        return X, U, np.array([T])

    def _warm_start(self, prev: Plan, x0: NDArray, t0: float):
        T = max(prev.t_end - t0, self.params.p_min + 0.5)
        tk = t0 + T * self.tau
        X = np.stack([prev.x_at(t) for t in tk], axis=1)
        U = np.stack([prev.u_at(t) for t in tk], axis=1)
        # blend the initial deviation out over the first part of the trajectory
        dev = x0 - X[:, 0]
        w = np.clip(1.0 - self.tau / 0.3, 0.0, 1.0)
        X = X + dev[:, None] * w[None]
        X[:, 0] = x0
        U[:, -1] = 0.0
        return X, U, np.array([T])

    # ================================================================================================ guess
    def _goal_heading_options(self, psi0: float, psi_goal: float) -> list[float]:
        base = psi0 + wrap_angle(psi_goal - psi0)
        other = base - 2 * np.pi * np.sign(base - psi0) if abs(base - psi0) > 1e-6 else base + 2 * np.pi
        return [base, other]

    def initial_guess(self, x0: Optional[NDArray] = None, goal_state=None, t0: float = 0.0, time_scale: float = 1.0):
        """
        Initial guess for SCvx: collision-free geometric path (grid search around static obstacles,
        through a pre-docking point on the dock axis), trapezoidal speed profile and smooth heading.
        """
        K = self.K
        if x0 is None or goal_state is None:
            return np.zeros((self.n_x, K)), np.zeros((self.n_u, K)), np.ones(self.n_p)
        P = self.params
        goal_xy = np.array([goal_state.x, goal_state.y])
        path = self._guess_path(x0[:2], goal_xy, goal_state.psi)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = s_cum[-1]
        v_c, a_c = 1.6, 0.4
        if L >= v_c**2 / a_c:
            T = L / v_c + v_c / a_c
        else:
            T = 2 * np.sqrt(max(L, 1e-3) / a_c)
        T = (T + 2.0 + 0.3 * abs(goal_state.psi - x0[2])) * time_scale
        T = float(np.clip(T, 3.0, max(P.t_max - t0, 3.5)))
        t = T * self.tau
        s = _trapezoid_profile(t, T, L)
        xy = np.stack([np.interp(s, s_cum, path[:, 0]), np.interp(s, s_cum, path[:, 1])], axis=0)
        vw = np.gradient(xy, t, axis=1)
        vw[:, 0] = 0.0
        vw[:, -1] = 0.0
        sm = 3 * self.tau**2 - 2 * self.tau**3
        psi = x0[2] + (goal_state.psi - x0[2]) * sm
        dpsi = np.gradient(psi, t)
        dpsi[-1] = 0.0
        c, sn = np.cos(psi), np.sin(psi)
        vx_b = c * vw[0] + sn * vw[1]
        vy_b = -sn * vw[0] + c * vw[1]
        X = np.zeros((self.n_x, K))
        X[0:2] = xy
        X[2] = psi
        X[3] = np.clip(vx_b, *self.vel_lim)
        X[4] = np.clip(vy_b, *self.vel_lim)
        X[5] = dpsi
        X[6] = 0.0
        X[7] = np.maximum(x0[7] - 0.1 * self.tau, self.sp.m_v + 1e-3)
        X[:, 0] = x0
        X[3:5, -1] = [goal_state.vx, goal_state.vy]
        U = np.zeros((self.n_u, K))
        return X, U, np.array([T])

    def _build_grid(self):
        P = self.params
        if self.boundary_corners is not None:
            lo = self.boundary_corners.min(axis=0)
            hi = self.boundary_corners.max(axis=0)
        else:
            lo, hi = np.array([-15.0, -15.0]), np.array([15.0, 15.0])
        res = P.grid_res
        xs = np.arange(lo[0] + res / 2, hi[0], res)
        ys = np.arange(lo[1] + res / 2, hi[1], res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        clear = np.full(len(pts), np.inf)
        for pl in self.planets.values():
            clear = np.minimum(clear, np.linalg.norm(pts - np.array(pl.center), axis=1) - pl.radius)
        if self.dock is not None:
            a, b = dock_segments(self.dock)
            for ai, bi in zip(a, b):
                ab = bi - ai
                lam = np.clip(((pts - ai) @ ab) / (ab @ ab), 0, 1)
                clear = np.minimum(clear, np.linalg.norm(pts - (ai + lam[:, None] * ab), axis=1) - 0.05)
        if self.boundary_corners is not None:
            n, w = boundary_halfplanes(self.boundary_corners)
            for ni, wi in zip(n, w):
                clear = np.minimum(clear, wi - pts @ ni)
        self._grid_lo, self._grid_res = lo, res
        self._grid_shape = gx.shape
        self._grid_pts = pts
        self._grid_clear = clear
        free = clear >= P.guess_clearance
        # 8-connected graph over free cells
        nx_, ny_ = gx.shape
        idx = np.arange(nx_ * ny_).reshape(nx_, ny_)
        rows, cols, wts = [], [], []
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            # pairs (i, j) -> (i + dx, j + dy), dx >= 0
            ja = slice(max(0, -dy), ny_ - max(0, dy))
            jb = slice(max(0, dy), ny_ - max(0, -dy))
            a = idx[0 : nx_ - dx, ja]
            b = idx[dx:nx_, jb]
            a, b = a.ravel(), b.ravel()
            ok = free[a] & free[b]
            rows.append(a[ok])
            cols.append(b[ok])
            wts.append(np.full(ok.sum(), res * np.hypot(dx, dy)))
        rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        n_nodes = nx_ * ny_
        self._graph = coo_matrix((wts, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        self._grid_free = free

    def _nearest_free(self, xy: NDArray) -> int:
        d = np.linalg.norm(self._grid_pts - xy, axis=1)
        d[~self._grid_free] = np.inf
        return int(np.argmin(d))

    def _segment_free(self, a: NDArray, b: NDArray, clearance: float) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / (0.5 * self._grid_res)) + 1)
        pts = a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]
        ij = np.floor((pts - self._grid_lo) / self._grid_res).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, self._grid_shape[0] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, self._grid_shape[1] - 1)
        flat = ij[:, 0] * self._grid_shape[1] + ij[:, 1]
        return bool(np.all(self._grid_clear[flat] >= clearance))

    def _guess_path(self, start: NDArray, goal: NDArray, goal_psi: float) -> NDArray:
        P = self.params
        end = goal
        tail = []
        if self.dock is not None:
            axis = np.array([np.cos(goal_psi), np.sin(goal_psi)])
            for d in (P.predock_dist, 2.5, 2.0, 3.5, 4.0):
                pre = goal + d * axis
                ij = self._nearest_free(pre)
                if np.linalg.norm(self._grid_pts[ij] - pre) < 2 * self._grid_res:
                    break
            end = pre
            tail = [goal]
        i0, i1 = self._nearest_free(start), self._nearest_free(end)
        _, pred = dijkstra(self._graph, directed=False, indices=i0, return_predecessors=True)
        if pred[i1] < 0 and i1 != i0:
            pts = [start, end]
        else:
            chain = [i1]
            while chain[-1] != i0:
                chain.append(pred[chain[-1]])
            chain.reverse()
            pts = [start] + [self._grid_pts[i] for i in chain] + [end]
        pts = np.array(pts)
        # string pulling
        out = [pts[0]]
        i = 0
        clr = 0.8 * P.guess_clearance
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._segment_free(pts[i], pts[j], clr):
                j -= 1
            out.append(pts[j])
            i = j
        out += tail
        return np.array(out)


# ==================================================================================================== utils
def _with_psi(goal_state: DynObstacleState, psi: float) -> DynObstacleState:
    return DynObstacleState(
        x=goal_state.x, y=goal_state.y, psi=psi, vx=goal_state.vx, vy=goal_state.vy, dpsi=goal_state.dpsi
    )


def _trapezoid_profile(t: NDArray, T: float, L: float) -> NDArray:
    """Arc length s(t) of a symmetric trapezoidal velocity profile covering L in time T (accel phase T/4)."""
    ta = T / 4.0
    vmax = L / (T - ta)
    a = vmax / ta
    s = np.where(
        t < ta,
        0.5 * a * t**2,
        np.where(t < T - ta, 0.5 * a * ta**2 + vmax * (t - ta), L - 0.5 * a * (T - t) ** 2),
    )
    return np.clip(s, 0.0, L)


def _better(a: Plan, b: Plan) -> bool:
    """Feasible beats infeasible; then lower cost; infeasible plans compared by violation."""
    if a.feasible != b.feasible:
        return a.feasible
    if a.feasible:
        return a.cost < b.cost
    return (a.max_violation + a.max_defect - min(a.min_clearance, 0)) < (
        b.max_violation + b.max_defect - min(b.min_clearance, 0)
    )
