import os
import random
import numpy as np

np.random.seed(574325)
from matplotlib import pyplot as plt

from advanced_genetic_operations import select_parents, uniform_crossover, gaussian_mutation, update_pareto_front, \
    polynomial_mutation, sbx_crossover, elitism_selection
from config import ALGORITHM_PARAMS, NO_OF_FILES, DATA_FILES_PATH, PORTFOLIO_PREFIX, FRONTIER_PREFIX, NO_OF_TRIALS
from data_parser import parse_portfolio_data, construct_covariance_matrix, read_frontier_data
from utils import initialize_population, evaluate_fitness, Individual

PARAMETER_RANGES = {
    'POP_SIZE': [20, 50, 100, 200, 250],  # Larger populations may explore the space better
    'NUM_GENERATIONS': [20, 50, 100, 150, 200, 250],  # More generations allow more time for convergence
    'CROSSOVER_RATE': [0.1, 0.3, 0.5, 0.4, 0.8, 1.0],  # Broader range to explore less and more frequent crossover
    'mutation_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0],  # Higher rates may escape local optima
    'mutation_shift': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0],  # Variation in the magnitude of mutation
    'ELITISM_COUNT': [1, 2, 3, 5, 10, 15, 20],  # Larger numbers of elites can stabilize progress towards the optimum
    'mutation_type': ['polynomial_mutation', 'gaussian_mutation'],  # Gaussian mutation could provide fine-tuned search
    'crossover_type': ['uniform_crossover', 'sbx_crossover']
}

crossover_fn_map = {
    "sbx_crossover": sbx_crossover,
    "uniform_crossover": uniform_crossover
}

mutation_fn_map = {
    "gaussian_mutation": gaussian_mutation,
    "polynomial_mutation": polynomial_mutation
}

def random_search(num_trials):
    best_score = float('-inf')
    best_params = None

    for i in range(num_trials):
        remaining_trials = num_trials - i
        params = {key: random.choice(value) for key, value in PARAMETER_RANGES.items()}
        print(f"Testing parameters: {params}")
        print(f"Trial {i + 1}/{num_trials} - Remaining trials: {remaining_trials - 1}")

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

pareto_checkpoints = []
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

        checkpoint_interval = self.params['NUM_GENERATIONS'] // 5  # 20% intervals
        

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
                    offspring1_weights, offspring2_weights = crossover_fn_map[self.params["crossover_type"]](
                        parent1.weights,
                        parent2.weights)
                else:
                    offspring1_weights, offspring2_weights = parent1.weights.copy(), parent2.weights.copy()
                mutation_fn = mutation_fn_map[self.params["mutation_type"]]
                offspring1 = Individual(mutation_fn(offspring1_weights))
                offspring2 = Individual(mutation_fn(offspring2_weights))
                new_population.extend([offspring1, offspring2])

            if generation % checkpoint_interval == 0 or generation == self.params['NUM_GENERATIONS'] - 1:
                pareto_checkpoints.append((generation, list(pareto_front)))


            population = new_population

        return pareto_front

    def run_ga(self, do_plot_correlation=True):
        pareto_front = self.optimize()
        self.plot_results(pareto_front)
        if (do_plot_correlation):
            self.plot_correlation(pareto_front)

    def plot_results(self, pareto_front):
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        pareto_front_returns = [ind.fitness[0] for ind in pareto_front]
        pareto_front_variances = [-ind.fitness[1] for ind in pareto_front]
        ax.scatter(pareto_front_returns, pareto_front_variances, c='b', marker='o', label='Pareto Front')
        frontier_points = read_frontier_data(self.frontier_file)
        frontier_returns, variances = zip(*frontier_points)  # Assuming each point is a tuple of (return, variance)
        ax.scatter(frontier_returns, variances, c='r', marker='x', label='Efficient Frontier')

        ax.set_xlabel('Return')
        ax.set_ylabel('Variance')
        ax.set_title('Comparison of GA Pareto Front with Efficient Frontier')
        ax.legend()
        ax.grid(True)
        plt.show()

    def plot_progress(self, checkpoints):
        for (generation, pareto_front) in checkpoints:
            # Plotting code as per your 'plot_results' method, modified to include generation info.
            pareto_front_returns = [ind.fitness[0] for ind in pareto_front]
            pareto_front_variances = [-ind.fitness[1] for ind in pareto_front]
            plt.scatter(pareto_front_returns, pareto_front_variances, label=f'Gen {generation}')
        
        plt.title('Pareto Front Progression')
        plt.xlabel('Return')
        plt.ylabel('Variance')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_correlation(self, pareto_front):
        frontier_points = read_frontier_data(self.frontier_file)
        # Calculate correlations between each point in the Pareto front and each point on the efficient frontier
        correlations = np.zeros((len(pareto_front), len(frontier_points)))
        for i, pareto_point in enumerate(pareto_front):
            for j, frontier_point in enumerate(frontier_points):
                pareto_return, pareto_variance = pareto_point.fitness
                frontier_return, frontier_variance = frontier_point
                correlations[i, j] = np.corrcoef([pareto_return, frontier_return],
                                                 [pareto_variance, frontier_variance])[0, 1]

        # Visualize correlations as a heatmap
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        cax = ax.matshow(correlations, cmap='coolwarm')
        plt.colorbar(cax)

        ax.set_xlabel('Pareto Front Returns')
        ax.set_ylabel('Efficient Frontier Returns')
        # Remove x-axis and y-axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Correlation between Pareto Front and Efficient Frontier')
        plt.show()


if __name__ == '__main__':
    user_input = input("Do you want to find the best parameters? (yes/no): ").strip().lower()
    if user_input == 'yes':
        best_params = random_search(NO_OF_TRIALS)  # Perform trials
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
            optimizer.plot_progress(pareto_checkpoints)  # Plot the progress

