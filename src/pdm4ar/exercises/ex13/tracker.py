"""
Time-varying LQR tracking of an SCvx plan, with disturbance estimation and feed-forward.

* The plan is re-sampled at the simulator command period (0.1 s). Around each reference point the nominal
  dynamics are linearised and discretised exactly for a zero-order-hold input (how the simulator applies
  commands).
* The state is augmented with a constant external disturbance d = [a_x, a_y, alpha] (world-frame linear
  acceleration and angular acceleration, the structure of the simulator's disturbance burst). A finite-horizon
  backward Riccati recursion on the augmented system gives gains u = u_ref - K_x e - K_d d_hat, i.e. LQR feedback
  plus the optimal feed-forward compensation of the estimated disturbance.
* d_hat is estimated from the one-step prediction residual of the nominal model (the simulator is deterministic,
  so the residual is essentially the disturbance itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from pdm4ar.exercises.ex13.discretization import BatchFirstOrderHold
from pdm4ar.exercises.ex13.geometry import wrap_angle
from pdm4ar.exercises.ex13.planner import Plan

N_D = 3  # disturbance dimension


@dataclass(frozen=True)
class TrackerParams:
    dt: float = 0.1
    # state weights: x, y, psi, vx, vy, dpsi, delta, m
    q: tuple = (6.0, 6.0, 8.0, 3.0, 3.0, 1.0, 0.2, 0.0)
    q_final: tuple = (30.0, 30.0, 30.0, 15.0, 15.0, 5.0, 0.5, 0.0)
    r: tuple = (1.0, 1.0)
    max_pos_err: float = 1.5  # error saturation (avoid huge commands far from the plan)
    max_psi_err: float = 0.8
    max_vel_err: float = 1.0
    use_disturbance_ff: bool = True
    dist_deadband: float = 0.01  # [m/s^2] estimates below this are treated as zero
    dist_tau: float = 0.5  # [s] the disturbance is modelled as decaying with this time constant in the LQR


def disturbance_input_matrix(psi: NDArray) -> NDArray:
    """B_d (N, 8, 3): world-frame acceleration and angular acceleration into the body-frame velocity dynamics."""
    c, s = np.cos(psi), np.sin(psi)
    Bd = np.zeros((len(psi), 8, N_D))
    Bd[:, 3, 0], Bd[:, 3, 1] = c, s
    Bd[:, 4, 0], Bd[:, 4, 1] = -s, c
    Bd[:, 5, 2] = 1.0
    return Bd


@dataclass
class TVLQRTracker:
    integrator: BatchFirstOrderHold
    thrust_limits: tuple
    ddelta_limits: tuple
    delta_limits: tuple
    params: TrackerParams = field(default_factory=TrackerParams)

    plan: Plan | None = None
    t_grid: NDArray | None = None
    x_ref: NDArray | None = None
    u_ref: NDArray | None = None
    gains: NDArray | None = None
    gains_d: NDArray | None = None

    # disturbance estimation
    d_hat: NDArray = field(default_factory=lambda: np.zeros(N_D))
    _last: tuple | None = None

    def set_plan(self, plan: Plan):
        P = self.params
        dt = P.dt
        n = max(1, int(np.ceil(plan.tf / dt - 1e-9)))
        t_grid = plan.t0 + dt * np.arange(n + 1)
        x_ref = np.stack([plan.x_at(t) for t in t_grid])
        # command applied on [t, t + dt]: use the average of the FOH plan input over the step
        u_ref = np.stack([0.5 * (plan.u_at(t) + plan.u_at(t + dt)) for t in t_grid])
        u_ref[-1] = 0.0
        _, A, B = self.integrator.jacobians_phys(x_ref, u_ref)
        Bdist = disturbance_input_matrix(x_ref[:, 2])
        nx, nu = A.shape[1], B.shape[2]
        nz = nx + N_D
        # exact ZOH discretisation of each (augmented) linearisation
        Mbig = np.zeros((nx + nu + N_D, nx + nu + N_D))
        Az = np.zeros((len(t_grid), nz, nz))
        Bz = np.zeros((len(t_grid), nz, nu))
        for k in range(len(t_grid)):
            Mbig[:nx, :nx] = A[k] * dt
            Mbig[:nx, nx : nx + nu] = B[k] * dt
            Mbig[:nx, nx + nu :] = Bdist[k] * dt
            E = expm(Mbig)
            Az[k, :nx, :nx] = E[:nx, :nx]
            Az[k, :nx, nx:] = E[:nx, nx + nu :]
            Az[k, nx:, nx:] = np.exp(-dt / P.dist_tau) * np.eye(N_D)
            Bz[k, :nx] = E[:nx, nx : nx + nu]
        Q = np.zeros((nz, nz))
        Q[:nx, :nx] = np.diag(P.q)
        S = np.zeros((nz, nz))
        S[:nx, :nx] = np.diag(P.q_final)
        R = np.diag(P.r)
        gains = np.zeros((len(t_grid), nu, nz))
        for k in range(len(t_grid) - 1, -1, -1):
            Ak, Bk = Az[k], Bz[k]
            BtS = Bk.T @ S
            Kk = np.linalg.solve(R + BtS @ Bk, BtS @ Ak)
            gains[k] = Kk
            S = Q + Ak.T @ S @ (Ak - Bk @ Kk)
            S = 0.5 * (S + S.T)
        self.plan = plan
        self.t_grid, self.x_ref, self.u_ref = t_grid, x_ref, u_ref
        self.gains, self.gains_d = gains[:, :, :nx], gains[:, :, nx:]

    # ------------------------------------------------------------------------------------------ estimation
    def observe(self, t: float, x: NDArray):
        """Update the disturbance estimate from the last applied command and the new measured state."""
        if self._last is not None:
            t_prev, x_prev, u_prev = self._last
            dt = t - t_prev
            if 1e-6 < dt < 0.3:
                x_pred = self.integrator.simulate_zoh(x_prev, lambda _t: u_prev, np.array([0.0, dt]))[-1]
                r = np.asarray(x, dtype=float) - x_pred
                psi = 0.5 * (x[2] + x_pred[2])
                c, s = np.cos(psi), np.sin(psi)
                a_body = r[3:5] / dt
                a_world = np.array([c * a_body[0] - s * a_body[1], s * a_body[0] + c * a_body[1]])
                d = np.array([a_world[0], a_world[1], r[5] / dt])
                d[np.abs(d) < self.params.dist_deadband] = 0.0
                self.d_hat = d
        self._last = None

    def record_command(self, t: float, x: NDArray, u: NDArray):
        self._last = (float(t), np.asarray(x, dtype=float).copy(), np.asarray(u, dtype=float).copy())

    # ------------------------------------------------------------------------------------------ control
    def reference(self, t: float) -> tuple[NDArray, NDArray, int]:
        k = int(np.clip(np.round((t - self.t_grid[0]) / self.params.dt), 0, len(self.t_grid) - 1))
        if t > self.t_grid[-1]:
            return self.x_ref[-1], np.zeros(2), len(self.t_grid) - 1
        return self.plan.x_at(t), self.u_ref[k], k

    def error(self, t: float, x: NDArray) -> NDArray:
        x_ref, _, _ = self.reference(t)
        e = np.asarray(x, dtype=float) - x_ref
        e[2] = wrap_angle(e[2])
        e[7] = 0.0
        return e

    def command(self, t: float, x: NDArray, d_hat: NDArray | None = None) -> NDArray:
        P = self.params
        _, u_ff, k = self.reference(t)
        e = self.error(t, x)
        # saturate the error to keep the linear controller in its validity region
        pe = np.linalg.norm(e[0:2])
        if pe > P.max_pos_err:
            e[0:2] *= P.max_pos_err / pe
        e[2] = np.clip(e[2], -P.max_psi_err, P.max_psi_err)
        ve = np.linalg.norm(e[3:5])
        if ve > P.max_vel_err:
            e[3:5] *= P.max_vel_err / ve
        u = u_ff - self.gains[k] @ e
        if P.use_disturbance_ff:
            d = self.d_hat if d_hat is None else d_hat
            # only compensate what the plan itself does not already account for
            u = u - self.gains_d[k] @ (d - self.plan.disturbance_at(t))
        u[0] = np.clip(u[0], *self.thrust_limits)
        u[1] = np.clip(u[1], *self.ddelta_limits)
        # do not push the nozzle beyond its limits (the simulator would zero the rate anyway)
        delta = x[6]
        if (delta >= self.delta_limits[1] and u[1] > 0) or (delta <= self.delta_limits[0] and u[1] < 0):
            u[1] = 0.0
        return u

    def rollout(
        self, x0: NDArray, t0: float, t_end: float, d_hat: NDArray | None = None, d_duration: float = 0.0, n_rk: int = 2
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Closed-loop simulation of this tracker on the nominal model (ZOH commands every dt, as in the simulator),
        optionally with the disturbance d_hat acting for d_duration seconds.
        Returns times (N,), states (N, 8) and the applied commands (N, 2).
        """
        dt = self.params.dt
        n = max(1, int(np.ceil((t_end - t0) / dt - 1e-9)))
        ts = t0 + dt * np.arange(n + 1)
        xs = np.zeros((n + 1, 8))
        us = np.zeros((n + 1, 2))
        x = np.asarray(x0, dtype=float).copy()
        f = self.integrator.f_phys
        h = dt / n_rk
        d_hat = np.zeros(N_D) if d_hat is None else np.asarray(d_hat, dtype=float)
        for i in range(n + 1):
            xs[i] = x
            if i == n:
                break
            active = ts[i] - t0 < d_duration
            d = d_hat if active else np.zeros(N_D)
            u = self.command(ts[i], x, d_hat=d)
            us[i] = u
            xr, ur = x[None], u[None]

            def fd(z):
                out = f(z, ur)
                if active:
                    out = out + disturbance_input_matrix(z[:, 2])[0] @ d
                return out

            for _ in range(n_rk):
                k1 = fd(xr)
                k2 = fd(xr + 0.5 * h * k1)
                k3 = fd(xr + 0.5 * h * k2)
                k4 = fd(xr + h * k3)
                xr = xr + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x = xr[0]
        return ts, xs, us
