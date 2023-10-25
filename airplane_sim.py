import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def as_vector(x=None, y=None, magnitude=None, angle=None):
    if magnitude is not None and angle is not None:
        x = magnitude * np.cos(angle)
        y = magnitude * np.sin(angle)
    elif x is None or y is None:
        raise ValueError("Must provide either x and y or magnitude and angle")
    return np.asarray([x, y])

def equations_of_motion(t, y, err_limit=100000):
    g = 9.81  # gravity, m/s^2
    rho = 1.225  # air density, kg/m^3
    CL_alpha = 2 * np.pi  # lift coefficient per radian
    wing_area = 0.01  # wing area, m^2
    mass = 0.01  # mass, kg
    inertia = 0.0001  # mass moment of inertia, kg*m^2
    cop_plane = as_vector(0.05, 0)  # position of center of pressure relative to the COM, relative to the plane, m
    critical_angle = np.deg2rad(90)  # critical angle of attack, radians

    cop_angle_plane = np.arctan2(*cop_plane)
    Fg = as_vector(0, mass * -g) # Force of gravity

    vx, vy, alpha, dalpha_dt = y[2:]
    speed = np.sqrt(vx**2 + vy**2)
    motion_angle = np.arctan2(vy, vx) # Angle of motion w.r.t to the x-axis
    attack_angle = alpha - motion_angle # Angle of attack w.r.t to the direction of motion

    # Lift and Drag calculations
    q = 0.5 * rho * speed**2 # Dynamic pressure
    CL = CL_alpha * np.sin(attack_angle) # Lift coefficient
    planform_area = wing_area * np.cos(attack_angle) # Projected wing area w.r.t to the direction of motion
    frontal_area = wing_area * np.sin(attack_angle) # Projected wing area w.r.t to the direction of motion

    # Limit max angle of attack. Too big, and the flow of air over the top of the wing will no longer be smooth and the lift suddenly decreases
    if np.abs(attack_angle) > critical_angle:
        lift = 0
    else:
        lift = q * planform_area * CL
    # TODO: Lift should be negative when attack angle is negative
    lift_vec = as_vector(magnitude=lift, angle=np.pi/2 - motion_angle) # Lift is perpendicular to the direction of motion

    # Aerodynamic forces
    CD = 0.01 + CL**2 / (np.pi * 4) # Drag coefficient
    drag = q * frontal_area * CD
    drag_vec = as_vector(magnitude=-drag, angle=motion_angle)
    pressure_forces = lift_vec + drag_vec
    # Total translational forces, in x-y reference frame
    F = pressure_forces + Fg
    dv_dt = F / mass

    # Torque and angular acceleration
    cop_angle = alpha + cop_angle_plane
    cop_arm = as_vector(magnitude=np.linalg.norm(cop_plane), angle=cop_angle) # COP arm position relative to the x-y reference frame
    tau = np.cross(cop_arm, pressure_forces) # Cross product of lift vector and distance from center of mass
    rotational_drag = 0.5 * rho * dalpha_dt**2 * wing_area * np.linalg.norm(cop_arm) * CD * np.sign(dalpha_dt)
    tau -= rotational_drag
    domega_dt = tau / inertia

    dstate_dt = [vx, vy, dv_dt[0], dv_dt[1], dalpha_dt, domega_dt]
    print(dstate_dt)
    if np.any(np.asarray(dstate_dt) > err_limit):
        raise RuntimeError(f"Things accelerated out of control. Reached {dstate_dt}")
    return dstate_dt


if __name__ == "__main__":
    # Initial conditions: [x_position, y_position, x_velocity, y_velocity, angle_of_attack, angular_velocity]
    angle = 0
    init_velocity = as_vector(magnitude=5, angle=np.deg2rad(angle))
    init_attack_angle = angle # degrees
    init_height = 1 # meters
    initial_conditions = [0, init_height, init_velocity[0], init_velocity[1], np.deg2rad(init_attack_angle), 0]

    duration = 5
    t = np.linspace(0, duration, 1001)
    solution = solve_ivp(equations_of_motion, (0, duration), initial_conditions, t_eval=t)

    x, y, vx, vy, alpha, omega = solution.y
    motion_angle = np.arctan2(vy, vx)
    above_ground = np.where(y >= 0)
    land_index = above_ground[0][-1]
    print(f"Landed after {t[land_index]} seconds")

    # Plotting
    plt.figure()
    plt.plot(x[above_ground], y[above_ground])
    # plt.plot(x[above_ground], alpha[above_ground])
    # plt.plot(t, x)
    # plt.plot(t, y)
    # plt.plot(x[above_ground], motion_angle[above_ground] - alpha[above_ground])
    arrow_size = 1/5
    for i in range(0, len(x[above_ground]), 10):
        plt.arrow(x[i], y[i], arrow_size*np.cos(alpha)[i], arrow_size*np.sin(alpha)[i], width=0.001)
    plt.title('Paper Airplane Flight')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.show()
