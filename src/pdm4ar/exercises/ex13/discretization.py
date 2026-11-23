import numpy as np
import sympy as spy
from numpy.typing import NDArray
from scipy.integrate import odeint

from pdm4ar.exercises.ex13.spaceship import SpaceshipDyn
from dg_commons.sim.models.spaceship import SpaceshipState, SpaceshipCommands
from pdm4ar.exercises_def.ex13.disturbance import FuelSpaceshipModel


class DiscretizationMethod:
    K: int  # number of discretization points
    N_sub: int  # number of substeps to approximate the ode
    range_t: tuple  # range of discretization points

    spaceship: SpaceshipDyn

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, spaceship: SpaceshipDyn, K: int, N_sub: int):
        # number of discretization points
        self.K = K

        # tuple containing N_sub ts between [0, 1/(K-1)] used to approximate the ODE
        self.N_sub = N_sub
        self.range_t = tuple(np.linspace(0, 1.0 / (self.K - 1), self.N_sub))

        self.spaceship = spaceship
        self.f, self.A, self.B, self.F = spaceship.get_dynamics()

        # number of states, inputs and parameters
        self.n_x = spaceship.n_x
        self.n_u = spaceship.n_u
        self.n_p = spaceship.n_p

    def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Calculate discretization for given states, inputs and parameter matrices.

        :param X: Matrix of states at all time steps
        :param U: Matrix of inputs at all time steps
        :param p: Vector of parameters
        :return: The 2D flattened matrices for the discrete-time linear dynamics, required by the optimization framework
        """
        pass

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Piecewise integration of the continuous-time nonlinear dynamics, simulating the system response to the inputs U that
        produced the discrete-time linear state sequence X_l, starting from each state. Helpful to verify accuracy of linearization.

        :param X_l: Linear state evolution matrix
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The piecewise integrated dynamics
        """
        pass

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time, integrating the continuous-time nonlinear dynamics
        starting from the initial state x0 under the effect of the input U.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics
        """
        pass

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:
        """
        Simulate nonlinear behavior given an initial state and an input over time, integrating the continuous-time nonlinear dynamics
        starting from the initial state x0 under the effect of the input U.

        :param x0: Initial state
        :param U: Linear input evolution matrix
        :param p: Vector of parameters
        :return: The full integrated dynamics (with added points linked to N_subs)
        """
        pass

    def check_dynamics(self) -> bool:
        """Compare the symbolic dynamics with the dg-commons simulation model."""
        samples = (
            (SpaceshipState(-2.0, 1.0, 0.3, 0.4, -0.2, 0.1, 0.2, 2.5), SpaceshipCommands(1.2, -0.1), 4.0),
            (SpaceshipState(3.0, -1.0, -0.7, -0.3, 0.25, -0.15, -0.3, 2.3), SpaceshipCommands(-0.8, 0.2), 2.5),
        )
        for state, command, final_time in samples:
            parameters = np.array([final_time])
            model = FuelSpaceshipModel(state, self.spaceship.sg, self.spaceship.sp)
            expected = final_time * model.dynamics(state, command).as_ndarray()
            actual = np.asarray(self.f(state.as_ndarray(), command.as_ndarray(), parameters), dtype=float).reshape(-1)
            if not np.allclose(actual, expected, rtol=1e-7, atol=1e-9):
                return False
        return True


class ZeroOrderHold(DiscretizationMethod):
    A_bar: NDArray
    B_bar: NDArray
    F_bar: NDArray
    r_bar: NDArray

    x_ind: slice
    A_bar_ind: slice
    B_bar_ind: slice
    F_bar_ind: slice
    r_bar_ind: slice

    P0: NDArray

    def __init__(self, spaceship: SpaceshipDyn, K: int, N_sub: int):

        super().__init__(spaceship, K, N_sub)

        # x+ = A_bar(x*(k))x(k) + B_bar(x*(k))u(k) + F_bar(x*(k))p + r_bar(k)
        self.A_bar = np.zeros([self.n_x * self.n_x, self.K - 1])
        self.B_bar = np.zeros([self.n_x * self.n_u, self.K - 1])
        self.F_bar = np.zeros([self.n_x * self.n_p, self.K - 1])
        self.r_bar = np.zeros([self.n_x, self.K - 1])

        # vector indices for flat matrices
        x_end = self.n_x
        A_bar_end = self.n_x * (1 + self.n_x)
        B_bar_end = self.n_x * (1 + self.n_x + self.n_u)
        F_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_p)
        r_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_p + 1)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.F_bar_ind = slice(B_bar_end, F_bar_end)
        self.r_bar_ind = slice(F_bar_end, r_bar_end)

        # integration initial condition
        self.P0 = np.zeros((self.n_x * (1 + self.n_x + self.n_u + self.n_p + 1),))
        self.P0[self.A_bar_ind] = np.eye(self.n_x).reshape(-1)

    def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:

        for k in range(self.K - 1):
            self.P0[self.x_ind] = X[:, k]
            P = np.array(odeint(self._ode_dPdt, self.P0, self.range_t, args=(U[:, k], p))[-1, :])

            Phi = P[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order="F")
            self.B_bar[:, k] = (Phi @ P[self.B_bar_ind].reshape((self.n_x, self.n_u))).flatten(order="F")
            self.F_bar[:, k] = (Phi @ P[self.F_bar_ind]).reshape((self.n_x, self.n_p)).flatten(order="F")
            self.r_bar[:, k] = Phi @ P[self.r_bar_ind]

        return self.A_bar, self.B_bar, self.F_bar, self.r_bar

    def _ode_dPdt(self, P: NDArray, t: float, u: NDArray, p: NDArray) -> NDArray:

        x = P[self.x_ind]

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(P[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        A_subs = self.A(x, u, p)
        B_subs = self.B(x, u, p)
        F_subs = self.F(x, u, p)
        f_subs = self.f(x, u, p).reshape(-1)

        dPdt = np.zeros_like(P)
        dPdt[self.x_ind] = f_subs.transpose()
        dPdt[self.A_bar_ind] = (A_subs @ P[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dPdt[self.B_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1)
        dPdt[self.F_bar_ind] = (Phi_A_xi @ F_subs).reshape(-1)
        r_t = f_subs - A_subs @ x - B_subs @ u - F_subs @ p
        dPdt[self.r_bar_ind] = Phi_A_xi @ r_t

        return dPdt

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_l[:, k], self.range_t, args=(U[:, k], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, self.K])
        X_nl[:, 0] = x0

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1] + (U.shape[1] - 1) * (self.N_sub - 2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], p))
            X_nl[:, k + 1] = x[-1, :]
            X_nl_dense[:, k * (self.N_sub - 1) + 1 : (k + 1) * (self.N_sub - 1) + 1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u: NDArray, p: NDArray) -> NDArray:
        return np.squeeze(self.f(x, u, p))



class FirstOrderHold(DiscretizationMethod):
    A_bar: NDArray
    B_plus_bar: NDArray
    B_minus_bar: NDArray
    F_bar: NDArray
    r_bar: NDArray

    x_ind: slice
    A_bar_ind: slice
    B_plus_bar_end: slice
    B_minus_bar_end: slice
    F_bar_ind: slice
    r_bar_ind: slice

    P0: NDArray

    def __init__(self, spaceship: SpaceshipDyn, K: int, N_sub: int):

        super().__init__(spaceship, K, N_sub)

        # x+ = A_bar(x*(k))x(k) + B_plus_bar(x*(k))u(k+1) + B_minus_bar(x*(k))u(k) + F_bar(x*(k))p + r_bar(k)
        self.A_bar = np.zeros([self.n_x * self.n_x, self.K - 1])
        self.B_plus_bar = np.zeros([self.n_x * self.n_u, self.K - 1])
        self.B_minus_bar = np.zeros([self.n_x * self.n_u, self.K - 1])
        self.F_bar = np.zeros([self.n_x * self.n_p, self.K - 1])
        self.r_bar = np.zeros([self.n_x, self.K - 1])

        # vector indices for flat matrices
        x_end = self.n_x
        A_bar_end = self.n_x * (1 + self.n_x)
        B_plus_bar_end = self.n_x * (1 + self.n_x + self.n_u)
        B_minus_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u)
        F_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u + self.n_p)
        r_bar_end = self.n_x * (1 + self.n_x + self.n_u + self.n_u + self.n_p + 1)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_plus_bar_ind = slice(A_bar_end, B_plus_bar_end)
        self.B_minus_bar_ind = slice(B_plus_bar_end, B_minus_bar_end)
        self.F_bar_ind = slice(B_minus_bar_end, F_bar_end)
        self.r_bar_ind = slice(F_bar_end, r_bar_end)

        # integration initial condition
        self.P0 = np.zeros((self.n_x * (1 + self.n_x + self.n_u + self.n_u + self.n_p + 1),))
        self.P0[self.A_bar_ind] = np.eye(self.n_x).reshape(-1)

    def calculate_discretization(
        self, X: NDArray, U: NDArray, p: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:

        for k in range(self.K - 1):
            self.P0[self.x_ind] = X[:, k]
            P = np.array(odeint(self._ode_dPdt, self.P0, self.range_t, args=(U[:, k], U[:, k + 1], p))[-1, :])

            # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi = P[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order="F")
            self.B_plus_bar[:, k] = (Phi @ P[self.B_plus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order="F")
            self.B_minus_bar[:, k] = (Phi @ P[self.B_minus_bar_ind].reshape((self.n_x, self.n_u))).flatten(order="F")
            self.F_bar[:, k] = (Phi @ P[self.F_bar_ind].reshape((self.n_x, self.n_p))).flatten(order="F")
            self.r_bar[:, k] = Phi @ P[self.r_bar_ind]

        return self.A_bar, self.B_plus_bar, self.B_minus_bar, self.F_bar, self.r_bar

    def _ode_dPdt(self, P: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:

        beta = (self.K - 1) * t
        alpha = 1 - beta
        x = P[self.x_ind]
        u = alpha * u_t0 + beta * u_t1

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(P[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        A_subs = self.A(x, u, p)
        B_subs = self.B(x, u, p)
        F_subs = self.F(x, u, p)
        f_subs = self.f(x, u, p).reshape(-1)

        dPdt = np.zeros_like(P)
        dPdt[self.x_ind] = f_subs.transpose()
        dPdt[self.A_bar_ind] = (A_subs @ P[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dPdt[self.B_plus_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * beta
        dPdt[self.B_minus_bar_ind] = (Phi_A_xi @ B_subs).reshape(-1) * alpha
        dPdt[self.F_bar_ind] = (Phi_A_xi @ F_subs).reshape(-1)

        r_t = f_subs - A_subs @ x - B_subs @ u - F_subs @ p

        dPdt[self.r_bar_ind] = Phi_A_xi @ r_t

        return dPdt

    def integrate_nonlinear_piecewise(self, X_l: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]

        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_l[:, k], self.range_t, args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl[:, 0] = x0

        for k in range(U.shape[1] - 1):
            X_nl[:, k + 1] = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], U[:, k + 1], p))[-1, :]

        return X_nl

    def integrate_nonlinear_full_dense(self, x0: NDArray, U: NDArray, p: NDArray) -> NDArray:

        X_nl = np.zeros([x0.size, U.shape[1]])
        X_nl_dense = np.zeros([x0.size, U.shape[1] + (U.shape[1] - 1) * (self.N_sub - 2)])
        X_nl[:, 0] = x0
        X_nl_dense[:, 0] = x0

        for k in range(U.shape[1] - 1):
            x = odeint(self._dxdt, X_nl[:, k], self.range_t, args=(U[:, k], U[:, k + 1], p))
            X_nl[:, k + 1] = x[-1, :]
            X_nl_dense[:, k * (self.N_sub - 1) + 1 : (k + 1) * (self.N_sub - 1) + 1] = np.array(x)[1:, :].T

        return X_nl_dense

    def _dxdt(self, x: NDArray, t: float, u_t0: NDArray, u_t1: NDArray, p: NDArray) -> NDArray:
        u = u_t0 + (self.K - 1) * t * (u_t1 - u_t0)
        return np.squeeze(self.f(x, u, p))


class BatchFirstOrderHold:
    """
    Vectorized first-order-hold discretization.

    Same linear model as :class:`FirstOrderHold`

        x[k+1] = A_k x[k] + B-_k u[k] + B+_k u[k+1] + F_k p + r_k

    but all K-1 intervals are integrated simultaneously with a fixed-step RK4 scheme on the
    variational (sensitivity) equations, which avoids per-interval ``odeint`` calls and matrix
    inversions. The Jacobians are hand-coded from the same equations as ``SpaceshipDyn`` (unit tests
    check them against the SymPy version). Matrices are returned as stacked arrays (K-1, n, m).
    """

    n_x = 8
    n_u = 2
    n_p = 1

    def __init__(self, l_r: float, Iz: float, C_T: float, K: int, N_sub: int = 5):
        self.l_r = float(l_r)
        self.Iz = float(Iz)
        self.C_T = float(C_T)
        self.K = K
        self.N_sub = N_sub  # RK4 steps per interval

    # ---------------------------------------------------------------- dynamics (physical time)
    def f_phys(self, x: NDArray, u: NDArray) -> NDArray:
        """x (N, 8), u (N, 2) -> dx/dt (N, 8)."""
        psi, vx, vy, w, d, m = x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7]
        F, ud = u[:, 0], u[:, 1]
        c, s = np.cos(psi), np.sin(psi)
        cd, sd = np.cos(d), np.sin(d)
        out = np.empty_like(x)
        out[:, 0] = vx * c - vy * s
        out[:, 1] = vx * s + vy * c
        out[:, 2] = w
        out[:, 3] = F / m * cd + w * vy
        out[:, 4] = F / m * sd - w * vx
        out[:, 5] = -self.l_r / self.Iz * F * sd
        out[:, 6] = ud
        out[:, 7] = -self.C_T * np.abs(F)
        return out

    def jacobians_phys(self, x: NDArray, u: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Returns f (N,8), A (N,8,8), B (N,8,2) in physical time."""
        N = x.shape[0]
        psi, vx, vy, w, d, m = x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7]
        F = u[:, 0]
        c, s = np.cos(psi), np.sin(psi)
        cd, sd = np.cos(d), np.sin(d)
        f = self.f_phys(x, u)
        A = np.zeros((N, 8, 8))
        A[:, 0, 2] = -vx * s - vy * c
        A[:, 0, 3] = c
        A[:, 0, 4] = -s
        A[:, 1, 2] = vx * c - vy * s
        A[:, 1, 3] = s
        A[:, 1, 4] = c
        A[:, 2, 5] = 1.0
        A[:, 3, 4] = w
        A[:, 3, 5] = vy
        A[:, 3, 6] = -F / m * sd
        A[:, 3, 7] = -F / m**2 * cd
        A[:, 4, 3] = -w
        A[:, 4, 5] = -vx
        A[:, 4, 6] = F / m * cd
        A[:, 4, 7] = -F / m**2 * sd
        A[:, 5, 6] = -self.l_r / self.Iz * F * cd
        B = np.zeros((N, 8, 2))
        B[:, 3, 0] = cd / m
        B[:, 4, 0] = sd / m
        B[:, 5, 0] = -self.l_r / self.Iz * sd
        B[:, 6, 1] = 1.0
        B[:, 7, 0] = -self.C_T * np.sign(F)
        return f, A, B

    # ---------------------------------------------------------------- discretization
    def _deriv(self, x, Phi, Sm, Sp, Sf, u0, u1, beta, tf, dist=None):
        u = u0 + beta * (u1 - u0)
        f, A, B = self.jacobians_phys(x, u)
        if dist is not None:
            # additive world-frame acceleration (ax, ay) and angular acceleration, active on masked intervals
            d, active = dist
            c, s = np.cos(x[:, 2]), np.sin(x[:, 2])
            ax, ay, al = d[0] * active, d[1] * active, d[2] * active
            f = f.copy()
            f[:, 3] += c * ax + s * ay
            f[:, 4] += -s * ax + c * ay
            f[:, 5] += al
            A[:, 3, 2] += -s * ax + c * ay
            A[:, 4, 2] += -c * ax - s * ay
        At = tf * A
        Bt = tf * B
        dx = tf * f
        dPhi = At @ Phi
        dSm = At @ Sm + Bt * (1.0 - beta)
        dSp = At @ Sp + Bt * beta
        dSf = np.einsum("nij,nj->ni", At, Sf) + f
        return dx, dPhi, dSm, dSp, dSf

    def calculate_discretization(self, X: NDArray, U: NDArray, p: NDArray, dense: bool = False, disturbance=None):
        """
        :param X: (8, K) reference states, U: (2, K) reference inputs, p: (1,) final time
        :return: dict with A (K-1,8,8), Bm, Bp (K-1,8,2), F (K-1,8), r (K-1,8),
                 x_next (K-1,8) nonlinear propagation of each node over its interval and, if dense,
                 x_dense (K-1, N_sub+1, 8) intermediate nonlinear states.
        :param disturbance: optional (d (3,), duration [s]): constant world-frame acceleration and angular
                 acceleration acting from the start of the trajectory for `duration` seconds.
        """
        K = X.shape[1]
        n = K - 1
        tf = float(np.asarray(p).reshape(-1)[0])
        x = X[:, :-1].T.copy()
        u0 = U[:, :-1].T
        u1 = U[:, 1:].T
        Phi = np.repeat(np.eye(8)[None], n, axis=0)
        Sm = np.zeros((n, 8, 2))
        Sp = np.zeros((n, 8, 2))
        Sf = np.zeros((n, 8))
        h = 1.0 / (K - 1) / self.N_sub  # step in normalized time
        hb = 1.0 / self.N_sub  # step in interpolation coordinate beta
        xs = [x.copy()] if dense else None
        dist = None
        if disturbance is not None and disturbance[1] > 0 and np.any(disturbance[0]):
            t_start = tf * np.arange(n) / (K - 1)
            dist = (np.asarray(disturbance[0], dtype=float), (t_start < disturbance[1]).astype(float))
        for i in range(self.N_sub):
            b0 = i * hb
            k1 = self._deriv(x, Phi, Sm, Sp, Sf, u0, u1, b0, tf, dist)
            y2 = [v + 0.5 * h * d for v, d in zip((x, Phi, Sm, Sp, Sf), k1)]
            k2 = self._deriv(*y2, u0, u1, b0 + 0.5 * hb, tf, dist)
            y3 = [v + 0.5 * h * d for v, d in zip((x, Phi, Sm, Sp, Sf), k2)]
            k3 = self._deriv(*y3, u0, u1, b0 + 0.5 * hb, tf, dist)
            y4 = [v + h * d for v, d in zip((x, Phi, Sm, Sp, Sf), k3)]
            k4 = self._deriv(*y4, u0, u1, b0 + hb, tf, dist)
            x, Phi, Sm, Sp, Sf = [
                v + h / 6.0 * (a + 2 * b + 2 * c + d)
                for v, a, b, c, d in zip((x, Phi, Sm, Sp, Sf), k1, k2, k3, k4)
            ]
            if dense:
                xs.append(x.copy())
        r = (
            x
            - np.einsum("nij,nj->ni", Phi, X[:, :-1].T)
            - np.einsum("nij,nj->ni", Sm, u0)
            - np.einsum("nij,nj->ni", Sp, u1)
            - Sf * tf
        )
        out = {"A": Phi, "Bm": Sm, "Bp": Sp, "F": Sf, "r": r, "x_next": x}
        if dense:
            out["x_dense"] = np.stack(xs, axis=1)
        return out

    def simulate_zoh(self, x0: NDArray, u_of_t, t_grid: NDArray, n_rk: int = 4) -> NDArray:
        """Integrate the nominal model with piecewise-constant inputs on t_grid (like the simulator)."""
        xs = np.zeros((len(t_grid), 8))
        xs[0] = x0
        x = np.asarray(x0, dtype=float)[None].copy()
        for i in range(len(t_grid) - 1):
            dt = t_grid[i + 1] - t_grid[i]
            u = np.asarray(u_of_t(t_grid[i]), dtype=float)[None]
            h = dt / n_rk
            for _ in range(n_rk):
                k1 = self.f_phys(x, u)
                k2 = self.f_phys(x + 0.5 * h * k1, u)
                k3 = self.f_phys(x + 0.5 * h * k2, u)
                k4 = self.f_phys(x + h * k3, u)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            xs[i + 1] = x[0]
        return xs
