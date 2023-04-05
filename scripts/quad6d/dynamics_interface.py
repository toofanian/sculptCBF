from cbf_opt.dynamics import ControlAffineDynamics
import numpy as np


class QuadVerticalDynamics(ControlAffineDynamics):
    STATES = ["Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS = ["T1", "T2"]
    PERIODIC_DIMS = [2]

    def __init__(self, params, **kwargs):
        self.Cd_v = params["Cd_v"]
        self.g = params["g"]
        self.Cd_phi = params["Cd_phi"]
        self.mass = params["mass"]
        self.length = params["length"]
        self.Iyy = params["Iyy"]
        super().__init__(params, **kwargs)

    def open_loop_dynamics(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        f = np.zeros_like(state)
        f[..., 0] = state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1] - self.g
        f[..., 2] = state[..., 3]
        f[..., 3] = -self.Cd_phi / self.Iyy * state[..., 3]
        return f

    def control_matrix(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, 0] = np.cos(state[..., 2]) / self.mass
        B[..., 1, 1] = np.cos(state[..., 2]) / self.mass
        B[..., 3, 0] = -self.length / self.Iyy
        B[..., 3, 1] = self.length / self.Iyy
        return B

    def disturbance_jacobian(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state: np.ndarray, control: np.ndarray, time: float = 0) -> np.ndarray:
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] = 1.0
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 2] = -(control[..., 0] + control[..., 1]) * np.sin(state[..., 2]) / self.mass
        J[..., 2, 3] = 1.0
        J[..., 3, 3] = -self.Cd_phi / self.Iyy
        return J


class QuadPlanarDynamics(QuadVerticalDynamics):
    STATES = ["X", "XDOT", "Y", "YDOT", "PHI", "PHIDOT"]
    CONTROLS = ["T1", "T2"]
    PERIODIC_DIMS = [4]

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] = state[..., 1]
        f[..., 1] = -self.Cd_v / self.mass * state[..., 1]
        f[..., 2:] = super().open_loop_dynamics(state[..., 2:], time)
        return f

    def control_matrix(self, state: np.ndarray, time: float = 0) -> np.ndarray:
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, 0] = -1 / self.mass * np.sin(state[..., 4])
        B[..., 1, 1] = -1 / self.mass * np.sin(state[..., 4])
        B[..., 2:, :] = super().control_matrix(state[..., 2:], time)
        return B

    def state_jacobian(self, state: np.ndarray, control: np.ndarray, time: float = 0) -> np.ndarray:
        J = np.repeat(np.zeros_like(state)[..., None], state.shape[-1], axis=-1)
        J[..., 0, 1] = 1
        J[..., 1, 1] = -self.Cd_v / self.mass
        J[..., 1, 4] = -1 / self.mass * (control[..., 0] + control[..., 1]) * np.cos(state[..., 4])
        J[..., 2:, 2:] = super().state_jacobian(state[..., 2:], control, time)
        return J


class QuadPlanarDynamicsInterface:
    def __init__(self):
        gravity: float = 9.81
        mass: float = 2.5
        Cd_v: float = 0.25
        drag_coefficient_phi: float = 0.02255
        length_between_copters: float = 1.0
        moment_of_inertia: float = 1.0

        self.u_min: float = 0
        self.u_max: float = 0.75 * mass * gravity
        self.dynamics = QuadPlanarDynamics(
            params={
                "Cd_v": Cd_v,
                "g": gravity,
                "Cd_phi": drag_coefficient_phi,
                "mass": mass,
                "length": length_between_copters,
                "Iyy": moment_of_inertia,
                "dt": 0.02,
            }
        )
