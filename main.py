import random

import numpy as np
from matplotlib import pyplot as plt

from advanced_genetic_operations import sbx_crossover, polynomial_mutation
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

pop_size = 100
num_assets = 10
initial_population = initialize_population(pop_size, num_assets)


# Function to evaluate the fitness of each portfolio
def evaluate_population(population, returns, cov_matrix):
    fitness = []
    for weights in population:
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(np.dot(weights, cov_matrix), weights)
        fitness.append((portfolio_return, portfolio_variance))
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



def mutate(individual):
    index = random.randint(0, len(individual) - 1)
    individual[index] = random.uniform(0, 1.0)
    individual /= individual.sum()  # Normalize to ensure sum equals 1.0
    return individual


# Load portfolio data
files = ["port1.txt", "port2.txt", "port3.txt", "port4.txt", "port5.txt"]
all_data = {}
for file_name in files:
    path = f"data/{file_name}"
    num_assets, returns, std_devs, correlations = parse_portfolio_data(path)
    cov_matrix = construct_covariance_matrix(num_assets, std_devs, correlations)
    all_data[file_name] = {"returns": returns, "cov_matrix": cov_matrix}

# Define algorithm parameters
pop_size = 100
num_assets = len(all_data[files[0]]['returns'])
num_generations = 50

# Initialize population
population = initialize_population(pop_size, num_assets)

# Evolutionary algorithm loop
for _ in range(num_generations):
    # Evaluate population
    fitness = evaluate_population(population, all_data[files[0]]['returns'], all_data[files[0]]['cov_matrix'])

    # Select parents based on Pareto dominance
    pareto_front = pareto_selection(population, fitness)

    # Perform crossover
    offspring = []
    while len(offspring) < pop_size:
        parent1, parent2 = random.sample(pareto_front, 2)
        child1, child2 = sbx_crossover(parent1, parent2)
        offspring.extend([child1, child2])

    # Perform mutation
    population = [polynomial_mutation(individual, 0.1) for individual in offspring]

# Extract Pareto front from the final population
pareto_front_fitness = evaluate_population(population, all_data[files[0]]['returns'], all_data[files[0]]['cov_matrix'])
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
