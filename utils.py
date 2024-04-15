import numpy as np

from Individual import Individual


def initialize_population(pop_size, num_assets):
    population = []
    for _ in range(pop_size):
        weights = np.random.rand(num_assets)
        weights /= np.sum(weights)  # Normalize to sum to 1
        population.append(Individual(weights))
    return population

def evaluate_fitness(individual, returns, cov_matrix):
    expected_return = calculate_expected_return(individual.weights, returns)
    portfolio_risk = calculate_portfolio_risk(individual.weights, cov_matrix)
    individual.fitness = [expected_return, -portfolio_risk]  # Maximize return, minimize risk


def calculate_expected_return(weights, returns):
    return np.dot(weights, returns)

def calculate_portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


