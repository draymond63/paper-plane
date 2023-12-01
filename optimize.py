import numpy as np
from airplane_sim import calc_distance_travelled, get_x1_bounds, get_x2_bounds, get_x3_bounds
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
import time

hyperparameter_sets = [
    {
        'max_num_iteration': 30,
        'population_size': 200,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.8,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None,
        'parents_portion': 0.3
    },
    {
        'max_num_iteration': 80,
        'population_size': 150,
        'mutation_probability': 0.2,
        'elit_ratio': 0.025,
        'crossover_probability': 0.7,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None,
        'parents_portion': 0.45
    },
    {
        'max_num_iteration': 20,
        'population_size': 50,
        'mutation_probability': 0.1,
        'elit_ratio': 0.025,
        'crossover_probability': 0.8,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None,
        'parents_portion': 0.3
    },
    {
        'max_num_iteration': 20,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None,
        'parents_portion': 0.3
    },
    {
        'max_num_iteration': 20,
        'population_size': 120,
        'mutation_probability': 0.1,
        'elit_ratio': 0.025,
        'crossover_probability': 0.8,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None,
        'parents_portion': 0.3
    }
]

def find_max_iteratively():
    lower_x3, upper_x3 = get_x3_bounds()
    lower_x2, upper_x2 = get_x2_bounds()
    data = []

    for x3 in np.linspace(lower_x3, upper_x3, 15):
        lower_x1, upper_x1 = get_x1_bounds(x3)
        for x2 in np.linspace(lower_x2, upper_x2, 30):
            for x1 in np.linspace(lower_x1, upper_x1, 10):
                print(f"{x1:.5f}, {x2:.5f}, {x3:.5f} ->", end=" ")
                m = calc_distance_travelled(x1, x2, x3)
                print(f"{m:.3f}")
                data.append([x1, x2, x3, m])

    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'distance_travelled'])

    # Save to CSV
    df.to_csv('output.csv', index=False)
    print("Data saved to 'output.csv'")

def wrapper_function(variables):
    x1, x2, x3 = variables

    # Update bounds of x1
    lower_x1, upper_x1 = get_x1_bounds(x3)
    
    # Check if variables are within bounds
    if lower_x1 <= x1 <= upper_x1 and lower_x2 <= x2 <= upper_x2 and lower_x3 <= x3 <= upper_x3:
        result = calc_distance_travelled(x1, x2, x3)
        return -result  # Since our problem is a maximize and genetic algo minimizes
    else:
        # Penalize solutions outside the bounds
        return np.inf

def genetic_algorithm():
    try:
        #np.random.seed(0)
        
        lower_x3, upper_x3 = get_x3_bounds()
        lower_x2, upper_x2 = get_x2_bounds()

        # Variable bounds for each variable 
        # From our constraints, x1 is smaller than W/(2+2√2) [where W is the width of the paper in meters]
        # From our calculations, 0.044714354 is the upper limit and the lower limit is set as 0 
        # [since a negative length isn't possible]
        varbound = np.array([[0, 0.044714354], [lower_x2, upper_x2], [lower_x3, upper_x3]])  

        # Set default hyperparameters
        algorithm_params = {
            'max_num_iteration': 20,
            'population_size': 120,
            'mutation_probability': 0.1,
            'elit_ratio': 0.025,
            'crossover_probability': 0.8,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None,
            'parents_portion': 0.3
        }
        model = ga(function=wrapper_function, dimension=3, variable_type='real', variable_boundaries=varbound,
               function_timeout=10, algorithm_parameters=algorithm_params)

        start_time = time.time()
        print("Start Time: ", start_time)
        # Run the optimization
        model.run()
        end_time = time.time()
        print("End Time: ", end_time)

        # Results
        solution = model.output_dict
        optimized_variables = solution['variable']
        optimized_objective = -solution['function']

        print("Optimized Variables:", optimized_variables)
        print("Optimized Objective:", optimized_objective)
        return solution, optimized_variables, optimized_objective, start_time, end_time
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        return None, None, None, None, None

def test_accuracy(model, optimized_objective, start_time, end_time):

    # True Objective is found through 
    true_objective = 66.79636033
    print(true_objective)

    # Calculate relative error
    relative_error = np.abs((true_objective - optimized_objective) / true_objective) * 100
    print('The Relative Error of this Algorithm is:', relative_error, '%')

    # Measure computational efficiency
    elapsed_time = end_time - start_time
    print(f"Computational Efficiency: {elapsed_time:.2f} seconds")

    # Define weights for relative error and computation time
    weight_relative_error = 0.7
    weight_computation_time = 0.3
    
    # Combine both metrics into a single score
    score = (weight_relative_error * relative_error) + (weight_computation_time * elapsed_time)
    print(f"Combined Score: {score:.2f}")

    return relative_error, elapsed_time, score


if __name__ == "__main__":
    lower_x3, upper_x3 = get_x3_bounds()
    lower_x2, upper_x2 = get_x2_bounds()

    # TO FIND ITERATIVE SOLUTION, UNCOMMENT
    #find_max_iteratively()

    solution, optimized_variables, optimized_objective, start_time, end_time = genetic_algorithm()

    #test_accuracy(solution, optimized_objective, start_time, end_time)

