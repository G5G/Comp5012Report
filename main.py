import os
import random
import numpy as np
np.random.seed(574325)
from matplotlib import pyplot as plt

from advanced_genetic_operations import select_parents, uniform_crossover, mutate, update_pareto_front, \
    polynomial_mutation, sbx_crossover, elitism_selection
from config import ALGORITHM_PARAMS, NO_OF_FILES, DATA_FILES_PATH, PORTFOLIO_PREFIX, FRONTIER_PREFIX
from data_parser import parse_portfolio_data, construct_covariance_matrix, read_frontier_data
from utils import initialize_population, evaluate_fitness, Individual

PARAMETER_RANGES = {
    'POP_SIZE': [20, 50, 100, 200, 250],  # Larger populations may explore the space better
    'NUM_GENERATIONS': [20, 50, 100, 150, 200, 250],  # More generations allow more time for convergence
    'CROSSOVER_RATE': [0.1, 0.3, 0.5, 0.4, 0.8, 1.0],  # Broader range to explore less and more frequent crossover
    'mutation_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0],  # Higher rates may escape local optima
    'mutation_shift': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0],  # Variation in the magnitude of mutation
    'ELITISM_COUNT': [1, 2, 3, 5, 10, 15, 20],  # Larger numbers of elites can stabilize progress towards the optimum
    'mutation_type': ['uniform', 'non-uniform', 'gaussian'],  # Gaussian mutation could provide fine-tuned search
    'crossover_type': ['one-point', 'two-point', 'uniform', 'sbx']  # Simulated Binary Crossover (SBX) is a good option to try
}




def random_search(num_trials):
    best_score = float('-inf')
    best_params = None

    for i in range(num_trials):
        remaining_trials = num_trials - i
        params = {key: random.choice(value) for key, value in PARAMETER_RANGES.items()}
        print(f"Testing parameters: {params}")
        print(f"Trial {i+1}/{num_trials} - Remaining trials: {remaining_trials - 1}")

        optimizer = PortfolioOptimization(
            f"{DATA_FILES_PATH}/{PORTFOLIO_PREFIX}1.txt",
            f"{DATA_FILES_PATH}/{FRONTIER_PREFIX}1.txt",
            params
        )
        pareto_front = optimizer.optimize()
        score = evaluate_pareto_front(pareto_front)
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New best score: {best_score} with params: {best_params}")

    return best_params


def evaluate_pareto_front(pareto_front):
    return np.mean([ind.fitness[0] - ind.fitness[1] for ind in pareto_front])

class PortfolioOptimization:
    def __init__(self, portfolio_file, frontier_file, params):
        self.portfolio_file = portfolio_file
        self.frontier_file = frontier_file
        self.params = params

    def optimize(self):
        num_assets, returns, std_devs, correlations = parse_portfolio_data(self.portfolio_file)
        cov_matrix = construct_covariance_matrix(num_assets, std_devs, correlations)
        population = initialize_population(self.params['POP_SIZE'], num_assets)
        pareto_front = []

        for generation in range(self.params['NUM_GENERATIONS']):
            elites = elitism_selection(population, self.params['ELITISM_COUNT'])
            new_population = elites[:]
            for individual in population:
                evaluate_fitness(individual, returns, cov_matrix)
            pareto_front = update_pareto_front(population, pareto_front)
            parents = select_parents(population, len(population))

            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i + 1]
                if np.random.rand() < self.params['CROSSOVER_RATE']:
                    offspring1_weights, offspring2_weights = uniform_crossover(parent1.weights, parent2.weights)
                else:
                    offspring1_weights, offspring2_weights = parent1.weights.copy(), parent2.weights.copy()

                offspring1 = Individual(mutate(offspring1_weights))
                offspring2 = Individual(mutate(offspring2_weights))
                new_population.extend([offspring1, offspring2])

            population = new_population

        return pareto_front

    def run_ga(self):
        pareto_front = self.optimize()
        self.plot_results(pareto_front)

    def plot_results(self, pareto_front):
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        pareto_front_returns = [ind.fitness[0] for ind in pareto_front]
        pareto_front_variances = [-ind.fitness[1] for ind in pareto_front]
        ax.scatter(pareto_front_returns, pareto_front_variances, c='b', marker='o', label='Pareto Front')
        frontier_points = read_frontier_data(self.frontier_file)
        returns, variances = zip(*frontier_points)
        ax.scatter(returns, variances, c='r', marker='x', label='Efficient Frontier')
        ax.set_xlabel('Return')
        ax.set_ylabel('Variance')
        ax.set_title('Comparison of GA Pareto Front with Efficient Frontier')
        ax.legend()
        ax.grid(True)
        plt.show()

if __name__ == '__main__':
    user_input = input("Do you want to find the best parameters? (yes/no): ").strip().lower()
    if user_input == 'yes':
        best_params = random_search(100)  # Perform 10 trials of random search
        print(f"Best parameters found: {best_params}")
        portfolio_file = f"{DATA_FILES_PATH}/{PORTFOLIO_PREFIX}1.txt"
        frontier_file = f"{DATA_FILES_PATH}/{FRONTIER_PREFIX}1.txt"
        optimizer = PortfolioOptimization(portfolio_file, frontier_file, best_params)
        optimizer.run_ga()
    else:
        for idx in range(NO_OF_FILES):
            portfolio_file = f"{DATA_FILES_PATH}/{PORTFOLIO_PREFIX}{idx + 1}.txt"
            frontier_file = f"{DATA_FILES_PATH}/{FRONTIER_PREFIX}{idx + 1}.txt"
            optimizer = PortfolioOptimization(portfolio_file, frontier_file, ALGORITHM_PARAMS)
            optimizer.run_ga()
