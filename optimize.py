import numpy as np
from airplane_sim import calc_distance_travelled, PAPER_WIDTH, PAPER_LENGTH



if __name__ == "__main__":
    for x1 in np.linspace(0.001, PAPER_WIDTH/2, 10):
        for x2 in np.linspace(0.001, PAPER_WIDTH/2, 10):
            for x3 in np.linspace(0.2, PAPER_LENGTH, 10):
                m = calc_distance_travelled(x1, x2, x3)
                print(f"{x1:.3f} {x2:.3f} {x3:.3f} -> {m:.3f}")
