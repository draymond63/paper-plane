import numpy as np
from airplane_sim import calc_distance_travelled, get_x1_bounds, get_x2_bounds, get_x3_bounds
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga


if __name__ == "__main__":
    def wrapper_function(variables):
        x1, x2, x3 = variables

        # Print the values of x1, x2, and x3 during optimization
        print("Current values: x1 =", x1, ", x2 =", x2, ", x3 =", x3)

        # Update the bounds for x1 based on the current value of x3
        lower_x1, upper_x1 = get_x1_bounds(x3)
    
        # Check if x1 is within the updated bounds
        if lower_x1 <= x1 <= upper_x1 and lower_x2 <= x2 <= upper_x2 and lower_x3 <= x3 <= upper_x3:
            # Your actual black-box function evaluation goes here
            result = calc_distance_travelled(x1, x2, x3)
            return -result  # Since geneticalgorithm minimizes, and you want to maximize
        else:
            # Penalize solutions outside the bounds
            return np.inf

    lower_x3, upper_x3 = get_x3_bounds()
    lower_x2, upper_x2 = get_x2_bounds()

    

    varbound = np.array([[0.03951667, 0.044714354], [lower_x2, upper_x2], [lower_x3, upper_x3]])  # Variable bounds for each variable
    population_size = 100  # Adjust this based on your needs
    initial_population = np.random.uniform(low=varbound[:, 0], high=varbound[:, 1], size=(population_size, varbound.shape[0]))

    algorithm_param = {'max_num_iteration': 20, 'population_size': 10, 'mutation_probability': 0.1, 'elit_ratio': 0.01,
                       'crossover_probability': 0.5, 'crossover_type': 'uniform', 'max_iteration_without_improv': None,
                        'population_size':100, 'parents_portion': 0.3}
    model = ga(function=wrapper_function, dimension=3, variable_type='real', variable_boundaries=varbound,
           function_timeout=10, algorithm_parameters=algorithm_param)

    # Run the optimization
    model.run()

    # Results
    solution = model.output_dict
    optimized_variables = solution['variable']
    optimized_objective = -solution['function']

    print("Optimized Variables:", optimized_variables)
    print("Optimized Objective:", optimized_objective)

    # iteration_per_variable = 10
    # lower_x3, upper_x3 = get_x3_bounds()
    # lower_x2, upper_x2 = get_x2_bounds()
    # data = []

    # for x3 in np.linspace(lower_x3, upper_x3, iteration_per_variable):
    #     lower_x1, upper_x1 = get_x1_bounds(x3)
    #     for x2 in np.linspace(lower_x2, upper_x2, iteration_per_variable):
    #         for x1 in np.linspace(lower_x1, upper_x1, iteration_per_variable):
    #             print(f"{x1:.5f}, {x2:.5f}, {x3:.5f} ->", end=" ")
    #             m = calc_distance_travelled(x1, x2, x3)
    #             print(f"{m:.3f}")
    #             data.append([x1, x2, x3, m])

    # df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'distance_travelled'])

    # # Save to CSV
    # df.to_csv('output.csv', index=False)
    # print("Data saved to 'output.csv'")

