import os
import random
import numpy as np
from matplotlib import pyplot as plt

from advanced_genetic_operations import select_parents, uniform_crossover, mutate, update_pareto_front
from config import ALGORITHM_PARAMS
from data_parser import parse_portfolio_data, construct_covariance_matrix, read_frontier_data
from utils import initialize_population, evaluate_fitness, Individual

def run_genetic_algorithm(portfolio_file, frontier_file):
    num_assets, returns, std_devs, correlations = parse_portfolio_data(portfolio_file)
    cov_matrix = construct_covariance_matrix(num_assets, std_devs, correlations)
    all_data = {"returns": returns, "cov_matrix": cov_matrix}

    pop_size = ALGORITHM_PARAMS['POP_SIZE']
    num_generations = ALGORITHM_PARAMS['NUM_GENERATIONS']
    crossover_rate = ALGORITHM_PARAMS['CROSSOVER_RATE']
    mutation_shift = ALGORITHM_PARAMS['mutation_shift']
    mutation_rate = ALGORITHM_PARAMS['mutation_rate']
    
    
    population = initialize_population(pop_size, num_assets)
    pareto_front = []

    for generation in range(num_generations):

        new_population = []

        for individual in population:
            evaluate_fitness(individual, returns, cov_matrix)
        
        pareto_front = update_pareto_front(population, pareto_front)

        parents = select_parents(population, pop_size)
        # simple uniform cross over
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            if np.random.rand() < crossover_rate:
                offspring1_weights, offspring2_weights = uniform_crossover(parent1.weights, parent2.weights)
            else:
                offspring1_weights, offspring2_weights = parent1.weights.copy(), parent2.weights.copy()

            offspring1 = Individual(mutate(offspring1_weights, mutation_rate, mutation_shift))
            offspring2 = Individual(mutate(offspring2_weights, mutation_rate, mutation_shift))
            new_population.extend([offspring1, offspring2])

        population = new_population




      
    return pareto_front, frontier_file

def plot_results(portfolio_file, frontier_file):
    pareto_front, frontier_data = run_genetic_algorithm(portfolio_file, frontier_file)
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # Plot GA results
    pareto_front_returns = [ind.fitness[0] for ind in pareto_front]  # Access the return part of fitness
    pareto_front_variances = [-ind.fitness[1] for ind in pareto_front]  # Access the variance part of fitness and negate it if needed
    
    ax.scatter(pareto_front_returns, pareto_front_variances, c='b', marker='o', label='Pareto Front')

    # Plot the efficient frontier
    frontier_points = read_frontier_data(frontier_file)
    returns, variances = zip(*frontier_points)  # Assuming each point is a tuple of (return, variance)
    ax.scatter(returns, variances, c='r', marker='x', label='Efficient Frontier')

    ax.set_xlabel('Return')
    ax.set_ylabel('Variance')
    ax.set_title(f'Comparison of GA Pareto Front with Efficient Frontier for {os.path.basename(portfolio_file)}')
    ax.legend()
    ax.grid(True)
    plt.show()


# Example usage:
plot_results('data/port1.txt', 'data/portef1.txt')
#plot_results('data/port2.txt', 'data/portef2.txt')
#plot_results('data/port3.txt', 'data/portef3.txt')
#plot_results('data/port4.txt', 'data/portef4.txt')
#plot_results('data/port5.txt', 'data/portef5.txt')

