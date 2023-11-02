import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field


def as_vector(x=None, y=None, magnitude=None, angle=None):
    if magnitude is not None and angle is not None:
        x = magnitude * np.cos(angle)
        y = magnitude * np.sin(angle)
    elif x is None or y is None:
        raise ValueError("Must provide either x and y or magnitude and angle")
    return np.asarray([x, y])


@dataclass
class Plane:
    wing_area: float = 0.01
    mass: float = 0.01
    inertia: float = 0.0001
    cop: np.ndarray = field(default_factory=lambda: np.asarray([0.001, 0]))
    "Center of pressure, relative to the center of mass. Positive x is forward, positive y is up"


class PlaneSim:
    def __init__(self) -> None:
        self.plane = Plane()
        self.g = 9.81  # gravity, m/s^2
        self.rho = 1.225  # air density, kg/m^3
        self.CL_alpha = 2 * np.pi  # lift coefficient per radian
        self.critical_angle = np.deg2rad(90)  # critical angle of attack, radians
        self.Fg = as_vector(0, self.plane.mass * -self.g) # Force of gravity
        self.cop_angle_plane = np.arctan2(self.plane.cop[1], self.plane.cop[0])
        self.cop_arm_length = np.linalg.norm(self.plane.cop)

    # TODO: Should the coefficient of lift be dependent on the angle of attack?
    def get_CL(self, attack_angle):
        return self.CL_alpha * np.sin(attack_angle)

    def get_CD(self, attack_angle):
        return 0.01 + self.get_CL(attack_angle)**2 / (np.pi * 4)

    def _ode(self, t, y, err_limit=1e9):
        vx, vy, alpha, dalpha_dt = y[2:]
        speed = np.sqrt(vx**2 + vy**2)
        motion_angle = np.arctan2(vy, vx) # Angle of motion w.r.t to the x-axis
        attack_angle = alpha - motion_angle # Angle of attack w.r.t to the direction of motion
        # Lift and Drag calculations
        Fp = np.sum(self.calc_pressure_forces(speed, motion_angle, attack_angle), axis=0)
        dv_dt = (Fp + self.Fg) / self.plane.mass
        # Torque and angular acceleration
        tau = self.calc_torque(alpha, dalpha_dt, attack_angle, Fp)
        domega_dt = tau / self.plane.inertia

        dstate_dt = [vx, vy, dv_dt[0], dv_dt[1], dalpha_dt, domega_dt]
        # if np.any(np.asarray(dstate_dt) > err_limit):
        #     raise RuntimeError(f"Things accelerated out of control. Reached {dstate_dt}")
        return dstate_dt

    def dynamic_pressure(self, speed):
        return 0.5 * self.rho * speed**2

    def calc_pressure_forces(self, speed, motion_angle, attack_angle):
        q = self.dynamic_pressure(speed) # Dynamic pressure
        planform_area = self.plane.wing_area * np.cos(attack_angle) # Projected wing area w.r.t to the direction of motion
        frontal_area = self.plane.wing_area * np.sin(attack_angle) # Projected wing area w.r.t to the direction of motion
        # Limit max angle of attack. Too big, and the flow of air over the top of the wing will no longer be smooth and the lift suddenly decreases
        lift = q * planform_area * self.get_CL(attack_angle) # if np.abs(attack_angle) <= self.critical_angle else 0
        lift_vec = as_vector(magnitude=lift, angle=np.pi/2 + motion_angle) # Lift is perpendicular to the direction of motion
        drag = q * frontal_area * self.get_CD(attack_angle)
        drag_vec = as_vector(magnitude=-drag, angle=motion_angle)
        return lift_vec, drag_vec

    def calc_torque(self, alpha, dalpha_dt, attack_angle, Fp):
        cop_arm = as_vector(magnitude=self.cop_arm_length, angle=alpha + self.cop_angle_plane) # COP arm position relative to the x-y reference frame
        tau = np.cross(cop_arm, Fp) # Cross product of lift vector and distance from center of mass
        tau -= self.calc_rotational_drag(dalpha_dt, attack_angle)
        return tau

    # Spin-damping moment
    def calc_rotational_drag(self, dalpha_dt, attack_angle):
        # TODO: Should dynamic pressure be a function of speed, not dalpha_dt?
        return np.sign(dalpha_dt) * self.dynamic_pressure(dalpha_dt) * self.plane.wing_area * self.cop_arm_length * self.get_CD(attack_angle)

    def run(self, t, height=2, speed=5, launch_angle=0, attack_angle=0):
        """
        Runs the simulation and returns the results as a tuple of numpy arrays

        Parameters
        ----------
        t : numpy array
            Time steps to run the simulation at
        height : float, optional
            Initial height of the plane (m)
        speed : float, optional
            Initial speed of the plane (m/s)
        launch_angle : float, optional
            Initial direction of travel (i.e. angle of velocity vector) (degrees, relative to the x-axis)
        attack_angle : float, optional
            Initial angle of attack of the plane (degrees, relative to the direction of travel)
        """
        init_velocity = as_vector(magnitude=speed, angle=np.deg2rad(launch_angle))
        initial_conditions = [0, height, *init_velocity, np.deg2rad(attack_angle + launch_angle), 0]

        duration = [*t][-1]
        solution = solve_ivp(self._ode, (0, duration), initial_conditions, t_eval=t)

        x, y, vx, vy, alpha, omega = solution.y
        above_ground = np.where(y >= 0)[0]
        land_index = above_ground[-1]
        landed_time = t[land_index]
        if landed_time == duration:
            raise RuntimeError(f"Didn't land. Started at height of {height} meters, ended at height of {y[land_index]:2f} meters")
        print(f"Landed after {t[land_index]} seconds")

        return solution.y

    def plot(self, results, with_forces=False, with_angle=True):
        x, y, vx, vy, alpha, omega = results
        above_ground = y >= 0
        plt.plot(x[above_ground], y[above_ground])
        legend = ['Flight path']
        if with_angle:
            direction_vecs = as_vector(magnitude=1, angle=alpha[above_ground])
            plt.quiver(x[above_ground], y[above_ground], direction_vecs[0], direction_vecs[1], width=0.001, scale=10, angles='xy', scale_units='xy')
            legend.append('Angle of attack')
        if with_forces:
            speed = np.sqrt(vx**2 + vy**2)
            motion_angle = np.arctan2(vy, vx)
            attack_angle = alpha - motion_angle
            lift, drag = self.calc_pressure_forces(speed, motion_angle, attack_angle)
            plt.quiver(x[above_ground], y[above_ground], lift[0][above_ground], lift[1][above_ground], width=0.001, scale=2, angles='xy', scale_units='xy', color='r')
            plt.quiver(x[above_ground], y[above_ground], drag[0][above_ground], drag[1][above_ground], width=0.001, scale=2, angles='xy', scale_units='xy', color='b')
            legend.extend(['Lift', 'Drag'])
        plt.title('Paper Airplane Flight')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.legend(legend)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    duration = 3
    t = np.linspace(0, duration, 201)
    sim = PlaneSim()
    sim.plot(sim.run(t), with_forces=True)
