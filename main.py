import os
import random

import numpy as np
from matplotlib import pyplot as plt

from advanced_genetic_operations import sbx_crossover, polynomial_mutation
from config import DATA_FILES_PATH, PORTFOLIO_PREFIX, ALGORITHM_PARAMS
from data_parser import parse_portfolio_data, construct_covariance_matrix


def initialize_population(pop_size, num_assets):
    """
    Initialize a population of portfolios.
    :param pop_size: Number of portfolios in the population
    :param num_assets: Number of assets in each portfolio
    :return: Initial population (numpy array)
    """
    population = np.random.rand(pop_size, num_assets)
    population /= population.sum(axis=1)[:, np.newaxis]  # Normalize to sum to 1
    return population


# Function to evaluate the fitness of each portfolio
def evaluate_population(population, all_data):
    fitness = []
    for weights in population:
        portfolio_returns = []
        portfolio_variances = []
        for file_name, data in all_data.items():
            returns = data['returns']
            cov_matrix = data['cov_matrix']
            if len(weights) != len(returns):
                continue  # Skip if dimensions don't match
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(np.dot(weights, cov_matrix), weights)
            portfolio_returns.append(portfolio_return)
            portfolio_variances.append(portfolio_variance)
        if portfolio_returns and portfolio_variances:  # Check if both returns and variances are calculated
            fitness.append((portfolio_returns, portfolio_variances))
    return fitness




# Function to perform selection based on Pareto dominance
def pareto_selection(population, fitness):
    pareto_front = []
    for i, (r1, v1) in enumerate(fitness):
        dominated = False
        for j, (r2, v2) in enumerate(fitness):
            if i != j and r2 >= r1 and v2 <= v1:
                dominated = True
                break
        if not dominated:
            pareto_front.append(population[i])
    return pareto_front

def integrate_elitism(population, fitness, elite_size=10):
    # Sort the population based on fitness; assuming smaller fitness values are better
    sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])
    elite_individuals = [ind for ind, fit in sorted_population[:elite_size]]
    return elite_individuals


all_data = {}
num_assets = 0

#track all the files name
files = []

for file_name in os.listdir(DATA_FILES_PATH):
    if file_name.startswith(PORTFOLIO_PREFIX) and file_name[len(PORTFOLIO_PREFIX)].isdigit():
        path = f"{DATA_FILES_PATH}/{file_name}"
        files.append(file_name)
        num_assets, returns, std_devs, correlations = parse_portfolio_data(path)
        cov_matrix = construct_covariance_matrix(num_assets, std_devs, correlations)
        all_data[file_name] = {"returns": returns, "cov_matrix": cov_matrix}

pop_size = ALGORITHM_PARAMS['POP_SIZE']
num_assets = len(all_data[files[0]]['returns'])
num_generations = ALGORITHM_PARAMS['NUM_GENERATIONS']

# Initialize population
population = initialize_population(pop_size, num_assets)

# Evolutionary algorithm loop
for i in range(num_generations):
    print(f"Generation {i}")

    # Evaluate population
    fitness = evaluate_population(population, all_data)

    # Integrate elitism: Extract elite individuals
    elite_individuals = integrate_elitism(population, fitness, elite_size=int(0.1 * len(population)))

    # Select parents based on Pareto dominance
    pareto_front = pareto_selection(population, fitness)

    # Perform crossover
    offspring = []
    while len(offspring) < pop_size:
        parent1, parent2 = random.sample(pareto_front, 2)
        child1, child2 = sbx_crossover(parent1, parent2)
        offspring.extend([child1, child2])

    # Ensure offspring list matches population size expected
    offspring = offspring[:pop_size - len(elite_individuals)]

    # Perform mutation on offspring only
    mutated_offspring = [polynomial_mutation(ind, 0.1) for ind in offspring]

    # New population formation with elitism
    population = elite_individuals + mutated_offspring

# Extract Pareto front from the final population
pareto_front_fitness = evaluate_population(population, all_data)
pareto_front = pareto_selection(population, pareto_front_fitness)

# Plot Pareto front
pareto_front_returns = [ind[0] for ind in pareto_front]
pareto_front_variances = [ind[1] for ind in pareto_front]
plt.figure(figsize=(8, 6))
plt.scatter(pareto_front_returns, pareto_front_variances, c='b', marker='o', label='Pareto front')
plt.xlabel('Return')
plt.ylabel('Variance')
plt.title('Pareto Front')
plt.legend()
plt.grid(True)
plt.show()
