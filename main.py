import os
import random

import numpy as np
from matplotlib import pyplot as plt

from advanced_genetic_operations import select_parents, uniform_crossover, mutate, update_pareto_front, \
    polynomial_mutation, sbx_crossover
from config import ALGORITHM_PARAMS, NO_OF_FILES, DATA_FILES_PATH, PORTFOLIO_PREFIX, FRONTIER_PREFIX
from data_parser import parse_portfolio_data, construct_covariance_matrix, read_frontier_data
from utils import initialize_population, evaluate_fitness, Individual


class PortfolioOptimization:
    def __init__(self, portfolio_file, frontier_file):
        self.portfolio_file = portfolio_file
        self.frontier_file = frontier_file
        self.mutation = [mutate, polynomial_mutation]
        self.crossover = [uniform_crossover, sbx_crossover]

    def optimize(self):
        num_assets, returns, std_devs, correlations = parse_portfolio_data(self.portfolio_file)
        cov_matrix = construct_covariance_matrix(num_assets, std_devs, correlations)

        pop_size = ALGORITHM_PARAMS['POP_SIZE']
        num_generations = ALGORITHM_PARAMS['NUM_GENERATIONS']
        crossover_rate = ALGORITHM_PARAMS['CROSSOVER_RATE']
        mutation_shift = ALGORITHM_PARAMS['mutation_shift']
        mutation_rate = ALGORITHM_PARAMS['mutation_rate']

        population = initialize_population(pop_size, num_assets)
        pareto_front = []

        for generation in range(num_generations):
            print(f'Generation {generation + 1}')
            new_population = []

            for individual in population:
                evaluate_fitness(individual, returns, cov_matrix)

            pareto_front = update_pareto_front(population, pareto_front)

            parents = select_parents(population, pop_size)

            # randomly select crossover and mutation
            mutate_fn = random.choice(self.mutation)
            crossover_fn = random.choice(self.crossover)
            # simple uniform cross over
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i + 1]
                if np.random.rand() < crossover_rate:
                    offspring1_weights, offspring2_weights = crossover_fn(parent1.weights, parent2.weights)
                else:
                    offspring1_weights, offspring2_weights = parent1.weights.copy(), parent2.weights.copy()

                offspring1 = Individual(mutate_fn(offspring1_weights, mutation_rate, mutation_shift))
                offspring2 = Individual(mutate_fn(offspring2_weights, mutation_rate, mutation_shift))
                new_population.extend([offspring1, offspring2])

            population = new_population

        return pareto_front

    def run_ga(self):
        pareto_front = self.optimize()
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)

        # Plot GA results
        pareto_front_returns = [ind.fitness[0] for ind in pareto_front]  # Access the return part of fitness
        pareto_front_variances = [-ind.fitness[1] for ind in
                                  pareto_front]  # Access the variance part of fitness and negate it if needed

        ax.scatter(pareto_front_returns, pareto_front_variances, c='b', marker='o', label='Pareto Front')

        # Plot the efficient frontier
        frontier_points = read_frontier_data(self.frontier_file)
        returns, variances = zip(*frontier_points)  # Assuming each point is a tuple of (return, variance)
        ax.scatter(returns, variances, c='r', marker='x', label='Efficient Frontier')

        ax.set_xlabel('Return')
        ax.set_ylabel('Variance')
        ax.set_title(
            f'Comparison of GA Pareto Front with Efficient Frontier for {os.path.basename(self.portfolio_file)}')
        ax.legend()
        ax.grid(True)
        plt.show()



if __name__ == '__main__':
    for idx in range(NO_OF_FILES):
        portfolio_optimizer = PortfolioOptimization(f"{DATA_FILES_PATH}/{PORTFOLIO_PREFIX}{idx + 1}.txt",
                                                    f"{DATA_FILES_PATH}/{FRONTIER_PREFIX}{idx + 1}.txt")
        portfolio_optimizer.run_ga()
