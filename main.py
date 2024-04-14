import os
import random

import numpy as np
from matplotlib import pyplot as plt

from advanced_genetic_operations import sbx_crossover, polynomial_mutation, pareto_selection, integrate_elitism
from config import DATA_FILES_PATH, PORTFOLIO_PREFIX, ALGORITHM_PARAMS
from data_parser import parse_portfolio_data, construct_covariance_matrix, load_portfolio_data
from utils import initialize_population, evaluate_population

all_data, files = load_portfolio_data(DATA_FILES_PATH, PORTFOLIO_PREFIX)

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
