import numpy as np
from airplane_sim import calc_distance_travelled, get_x1_bounds, get_x2_bounds, get_x3_bounds
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

def find_max_iteratively():
    iteration_per_variable = 30
    lower_x3, upper_x3 = get_x3_bounds()
    lower_x2, upper_x2 = get_x2_bounds()
    data = []

    for x3 in np.linspace(lower_x3, upper_x3, iteration_per_variable):
        lower_x1, upper_x1 = get_x1_bounds(x3)
        for x2 in np.linspace(lower_x2, upper_x2, iteration_per_variable):
            for x1 in np.linspace(lower_x1, upper_x1, iteration_per_variable):
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
    lower_x3, upper_x3 = get_x3_bounds()
    lower_x2, upper_x2 = get_x2_bounds()

    # Variable bounds for each variable 
    # From our constraints, x1 is smaller than W/(2+2√2) [where W is the width of the paper in meters]
    # From our calculations, 0.044714354 is the upper limit and the lower limit is set as 0 
    # [since a negative length isn't possible]
    varbound = np.array([[0, 0.044714354], [lower_x2, upper_x2], [lower_x3, upper_x3]])  
    
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
    return solution, optimized_variables, optimized_objective

def test_accuracy(model, optimized_objective):
    # True Objective is found through 
    true_objective = 65.42

    # Calculate relative error
    relative_error = np.abs((true_objective - optimized_objective) / true_objective) * 100
    print('The Relative Error of this Algorithm is:', relative_error, '%')
    
    # Plot convergence
    plt.plot(model.output_dict['output'])
    plt.title('Convergence Plot')
    plt.xlabel('Generation')
    plt.ylabel('Objective Function Value')
    plt.show()


if __name__ == "__main__":
    lower_x3, upper_x3 = get_x3_bounds()
    lower_x2, upper_x2 = get_x2_bounds()

    # TO FIND ITERATIVE SOLUTION UNCOMMENT
    #find_max_iteratively()

    solution, optimized_variables, optimized_objective = genetic_algorithm()
    test_accuracy(solution, optimized_objective)

