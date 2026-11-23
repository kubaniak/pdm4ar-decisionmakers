import sympy as spy

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters


class SpaceshipDyn:
    sg: SpaceshipGeometry
    sp: SpaceshipParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SpaceshipGeometry, sp: SpaceshipParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi delta m", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        extract the state from self.x the following way:
        0x 1y 2psi 3vx 4vy 5dpsi 6delta 7m
        """
        x, y, psi, vx, vy, dpsi, delta, m = self.x
        thrust, ddelta = self.u
        t_f = self.p[0]

        l_r = self.sg.l_r
        Iz = self.sg.Iz
        C_T = self.sp.C_T

        # Nominal dynamics in physical time (body-frame velocities, rotating-frame terms)
        f_phys = spy.Matrix(
            [
                vx * spy.cos(psi) - vy * spy.sin(psi),
                vx * spy.sin(psi) + vy * spy.cos(psi),
                dpsi,
                thrust / m * spy.cos(delta) + dpsi * vy,
                thrust / m * spy.sin(delta) - dpsi * vx,
                -l_r / Iz * thrust * spy.sin(delta),
                ddelta,
                -C_T * spy.Abs(thrust),
            ]
        )
        # Normalized time tau = t / t_f  ->  dX/dtau = t_f * f(X, U)
        f = t_f * f_phys

        # Jacobians and matrices of the system (don't need to change)
        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
