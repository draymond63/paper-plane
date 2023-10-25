import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def equations_of_motion(t, y):
    g = 9.81  # gravity, m/s^2
    rho = 1.225  # air density, kg/m^3
    CL_alpha = 2 * np.pi  # lift coefficient per radian
    wing_area = 0.01  # wing area, m^2
    mass = 0.01  # mass, kg
    inertia = 0.0001  # mass moment of inertia, kg*m^2
    # TODO: Include vertical position of center of mass and pressure
    COM = 0.05  # position of center of mass from tail edge, m
    COP = 0.1  # position of center of pressure from tail edge, m
    critical_angle = np.deg2rad(45)  # critical angle of attack, radians

    vx, vy, alpha, dalpha_dt = y[2:]
    speed = np.sqrt(vx**2 + vy**2)
    motion_angle = np.arctan2(vy, vx) # Angle of motion w.r.t to the x-axis
    attack_angle = alpha - motion_angle # Angle of attack w.r.t to the direction of motion

    # Lift and Drag calculations
    q = 0.5 * rho * speed**2  # dynamic pressure
    CL = CL_alpha * np.sin(attack_angle)  # Lift coefficient
    planform_area = wing_area * np.cos(attack_angle) # Projected wing area w.r.t to the direction of motion
    frontal_area = wing_area * np.sin(attack_angle) # Projected wing area w.r.t to the direction of motion

    # Limit max angle of attack. Too big, and the flow of air over the top of the wing will no longer be smooth and the lift suddenly decreases
    if np.abs(attack_angle) > critical_angle:
        lift = 0
        # CL = - 0.1
    else:
        lift = q * planform_area * CL
    # TODO: Lift should be negative when attack angle is negative
    lift_x = lift * np.cos(np.pi/2 - motion_angle)
    lift_y = lift * np.sin(np.pi/2 - motion_angle)

    # Aerodynamic forces
    CD = 0.01 + CL**2 / (np.pi * 4)
    drag = q * frontal_area * CD # TODO: Include friction drag?
    drag_x = drag * np.cos(motion_angle)
    drag_y = drag * np.sin(motion_angle)
    # Total translational forces
    Fx = -drag_x + lift_x
    Fy = -drag_y + lift_y - mass * g
    dvx_dt = Fx / mass
    dvy_dt = Fy / mass
    # Torque and angular acceleration
    tau = (COP - COM) * (lift * np.sin(attack_angle) + drag * np.cos(attack_angle)) # Cross product of lift vector and distance from center of mass
    # TODO: Should depend on the direction of the angular velocity
    rotational_drag = 0.5 * rho * dalpha_dt**2 * wing_area * (COP - COM) * CD * np.sign(dalpha_dt)
    tau -= rotational_drag
    domega_dt = tau / inertia

    return [vx, vy, dvx_dt, dvy_dt, dalpha_dt, domega_dt]

# Initial conditions: [x_position, y_position, x_velocity, y_velocity, angle_of_attack, angular_velocity]
launch_angle = 0 # degrees
init_attack_angle = 0 # degrees
init_height = 5 # meters
init_speed = 5 # m/s
initial_conditions = [0, init_height, init_speed*np.cos(np.deg2rad(launch_angle)), init_speed*np.sin(np.deg2rad(launch_angle)), np.deg2rad(init_attack_angle), 0]

# Time array from 0 to 5 seconds
duration = 5
t = np.linspace(0, duration, 1001)
# TODO: Plane stalls out and then the angle never recovers. Why?
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
