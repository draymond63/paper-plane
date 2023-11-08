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
    frontal_area: float = 0.01
    cop: np.ndarray = field(default_factory=lambda: np.asarray([0.01, 0.1]))
    "Center of pressure, relative to the center of mass. Positive x is forward, positive y is up"

    @classmethod
    def from_parameters(cls, x1, x2, x3):
        # TODO: Calculate inertia and wing area and mass from parameters
        return cls(wing_area=x1, mass=x2, inertia=x3)


class PlaneSim:
    def __init__(self, plane=Plane(), g=9.81, air_density=1.225) -> None:
        self.plane = plane
        self.g = g  # gravity, m/s^2
        self.rho = air_density  # air density, kg/m^3
        self.CL_alpha = 2 * np.pi  # lift coefficient per radian
        self.CD = 0.01 / (np.pi * 2)  # drag coefficient per radian
        self.critical_angle = np.deg2rad(90)  # critical angle of attack, radians
        self.Fg = as_vector(0, self.plane.mass * -self.g) # Force of gravity
        self.cop_angle_plane = np.arctan2(self.plane.cop[1], self.plane.cop[0])
        self.cop_arm_length = np.linalg.norm(self.plane.cop)

    def get_CL(self, attack_angle):
        return self.CL_alpha * np.sin(attack_angle)

    def _ode(self, t, y):
        vx, vy, alpha, dalpha_dt = y[2:]
        speed = np.linalg.norm([vx, vy], axis=0)
        motion_angle = np.arctan2(vy, vx) # Angle of motion w.r.t to the x-axis
        attack_angle = alpha - motion_angle # Angle of attack w.r.t to the direction of motion
        # Lift and Drag calculations
        Fp = np.sum(self.calc_pressure_forces(speed, motion_angle, attack_angle), axis=0)
        dv_dt = (Fp + self.Fg) / self.plane.mass
        # Torque and angular acceleration
        tau = self.calc_torque(alpha, dalpha_dt, Fp)
        domega_dt = tau / self.plane.inertia

        dstate_dt = [vx, vy, *dv_dt, dalpha_dt, domega_dt]
        return dstate_dt

    def dynamic_pressure(self, speed):
        return 0.5 * self.rho * speed**2

    def calc_pressure_forces(self, speed, motion_angle, attack_angle):
        q = self.dynamic_pressure(speed) # Dynamic pressure
        planform_area = self.plane.wing_area * np.cos(attack_angle) # Projected wing area w.r.t to the direction of motion
        frontal_area = self.plane.wing_area * np.abs(np.sin(attack_angle)) # Projected wing area w.r.t to the direction of motion
        frontal_area += self.plane.frontal_area * np.abs(np.cos(attack_angle)) # Add the frontal area of the plane
        # Limit max angle of attack. Too big, and the flow of air over the top of the wing will no longer be smooth and the lift suddenly decreases
        # TODO: Include critical angle or get rid of it
        lift = q * planform_area * self.get_CL(attack_angle) # if np.abs(attack_angle) <= self.critical_angle else 0
        lift_vec = as_vector(magnitude=lift, angle=np.pi/2 + motion_angle) # Lift is perpendicular to the direction of motion
        drag = q * frontal_area * self.CD
        drag_vec = as_vector(magnitude=-drag, angle=motion_angle)
        return lift_vec, drag_vec

    def calc_torque(self, alpha, dalpha_dt, Fp):
        cop_arm = as_vector(magnitude=self.cop_arm_length, angle=alpha + self.cop_angle_plane) # COP arm position relative to the x-y reference frame
        tau = np.cross(cop_arm, Fp) # Cross product of lift vector and distance from center of mass
        tau -= self.calc_rotational_drag(dalpha_dt)
        return tau

    # TODO: Should dynamic pressure be a function of speed, not dalpha_dt?
    # Spin-damping moment
    def calc_rotational_drag(self, dalpha_dt):
        return np.sign(dalpha_dt) * self.dynamic_pressure(dalpha_dt)**2 * self.plane.wing_area * self.cop_arm_length * self.CD

    # TODO: Make the plane a run parameter, not a initialization parameter
    def run(self, t, height=2, speed=5, launch_angle=0, attack_angle=0, must_land=True):
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
        below_ground = np.where(y <= 0)[0]
        if must_land and len(below_ground) == 0:
            raise RuntimeError(f"Didn't land. Started at height of {height} meters, ended at height of {y[-1]:2f} meters")
        elif len(below_ground) == 0:
            print("Didn't land")
        else:
            land_index = below_ground[0]
            print(f"Landed after {t[land_index]} seconds (travelled {x[land_index]:.2f} meters)")
        return solution.y

    def plot(self, results, with_forces=False, with_angle=True):
        x, y, vx, vy, alpha, omega = results
        below_ground = np.where(y <= 0)[0]
        t_end = below_ground[0] if len(below_ground) > 0 else len(y)
        plt.plot(x[:t_end], y[:t_end])
        legend = ['Flight path']
        if with_angle:
            direction_vecs = as_vector(magnitude=1, angle=alpha[:t_end])
            plt.quiver(x[:t_end], y[:t_end], direction_vecs[0], direction_vecs[1], width=0.001, scale=10, angles='xy', scale_units='xy')
            legend.append('Angle of attack')
        if with_forces:
            speed = np.linalg.norm([vx, vy], axis=0)
            motion_angle = np.arctan2(vy, vx)
            attack_angle = alpha - motion_angle
            lift, drag = self.calc_pressure_forces(speed, motion_angle, attack_angle)
            plt.quiver(x[:t_end], y[:t_end], lift[0][:t_end], lift[1][:t_end], width=0.001, angles='xy', scale_units='xy', color='r')
            plt.quiver(x[:t_end], y[:t_end] + 0.2, drag[0][:t_end], drag[1][:t_end], width=0.001, angles='xy', scale_units='xy', color='b')
            legend.extend(['Lift', 'Drag'])
        plt.title('Paper Airplane Flight')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.legend(legend)
        plt.grid(True)
        plt.show()


def calc_distance_travelled(plane: Plane, **flight_params):
    duration = 10 # TODO: Increase duration if error is raised
    t = np.linspace(0, duration, 201)
    sim = PlaneSim(plane)
    x, y, vx, vy, alpha, omega = sim.run(t, **flight_params)
    above_ground = y >= 0 # TODO: If plane swoops up back into the air, this will be wrong
    return x[above_ground][-1]


if __name__ == "__main__":
    duration = 8
    t = np.linspace(0, duration, 1001)
    sim = PlaneSim()
    sim.plot(sim.run(t, height=2, must_land=False), with_forces=True)
