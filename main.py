import os
import random
import numpy as np
from matplotlib import pyplot as plt

from advanced_genetic_operations import sbx_crossover, polynomial_mutation, pareto_selection, integrate_elitism
from config import DATA_FILES_PATH, PORTFOLIO_PREFIX, FRONTIER_PREFIX, ALGORITHM_PARAMS
from data_parser import parse_portfolio_data, construct_covariance_matrix, read_frontier_data
from utils import initialize_population, evaluate_population

def run_genetic_algorithm(portfolio_file, frontier_file):
    num_assets, returns, std_devs, correlations = parse_portfolio_data(portfolio_file)
    cov_matrix = construct_covariance_matrix(num_assets, std_devs, correlations)
    all_data = {"returns": returns, "cov_matrix": cov_matrix}

    pop_size = ALGORITHM_PARAMS['POP_SIZE']
    num_generations = ALGORITHM_PARAMS['NUM_GENERATIONS']
    population = initialize_population(pop_size, num_assets)

    for i in range(num_generations):
        fitness = evaluate_population(population, all_data)
        elite_individuals = integrate_elitism(population, fitness, elite_size=int(0.1 * len(population)))
        pareto_front = pareto_selection(population, fitness)
        offspring = []
        while len(offspring) < pop_size:
            parent1, parent2 = random.sample(pareto_front, 2)
            child1, child2 = sbx_crossover(parent1, parent2)
            offspring.extend([child1, child2])
        offspring = offspring[:pop_size - len(elite_individuals)]
        mutated_offspring = [polynomial_mutation(ind, 0.1) for ind in offspring]
        population = elite_individuals + mutated_offspring

    pareto_front_fitness = evaluate_population(population, all_data)
    final_pareto_front = pareto_selection(population, pareto_front_fitness)
    return final_pareto_front, frontier_file

def plot_results(portfolio_file, frontier_file):
    pareto_front, frontier_data = run_genetic_algorithm(portfolio_file, frontier_file)
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # Plot GA results
    pareto_front_returns = [ind[0] for ind in pareto_front]
    pareto_front_variances = [ind[1] for ind in pareto_front]
    ax.scatter(pareto_front_returns, pareto_front_variances, c='b', marker='o', label='Pareto Front')

    # Plot the efficient frontier
    frontier_points = read_frontier_data(frontier_file)
    returns, variances = zip(*frontier_points)
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

