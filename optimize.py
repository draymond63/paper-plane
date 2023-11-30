import numpy as np
from airplane_sim import calc_distance_travelled, get_x1_bounds, get_x2_bounds, get_x3_bounds


if __name__ == "__main__":
    iteration_per_variable = 10
    lower_x3, upper_x3 = get_x3_bounds()
    lower_x2, upper_x2 = get_x2_bounds()

    for x3 in np.linspace(lower_x3, upper_x3, iteration_per_variable):
        lower_x1, upper_x1 = get_x1_bounds(x3)
        for x2 in np.linspace(lower_x2, upper_x2, iteration_per_variable):
            for x1 in np.linspace(lower_x1, upper_x1, iteration_per_variable):
                print(f"{x1:.5f}, {x2:.5f}, {x3:.5f} ->", end=" ")
                m = calc_distance_travelled(x1, x2, x3)
                print(f"{m:.3f}")
