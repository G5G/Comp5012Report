

import numpy as np
import random
np.random.seed(574325)
random.seed(574325)
from config import ALGORITHM_PARAMS

def normalize_weights(weights):
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return weights
    else:
        return weights / sum_weights


def sbx_crossover(parent1, parent2, low=0, high=1):
    eta_c = ALGORITHM_PARAMS['SBX_ETA']
    rand = np.random.rand(parent1.shape[0])
    gamma = np.empty(parent1.shape[0])
    mask = rand <= 0.5
    
    gamma[mask] = (2 * rand[mask]) ** (1 / (eta_c + 1))
    gamma[~mask] = (1 / (2 * (1 - rand[~mask]))) ** (1 / (eta_c + 1))
    
    offspring1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
    offspring2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)
    
    offspring1 = normalize_weights(np.clip(offspring1, low, high))
    offspring2 = normalize_weights(np.clip(offspring2, low, high))
    
    return offspring1, offspring2

def polynomial_mutation(individual, low=0, high=1):
    mutation_rate = ALGORITHM_PARAMS['mutation_rate']
    eta_m = ALGORITHM_PARAMS['POLY_ETA']
    mutant = np.copy(individual)
    for i in range(len(mutant)):
        if np.random.rand() < mutation_rate:
            u = np.random.rand()
            delta = None
            if u < 0.5:
                delta_q = (2*u)**(1/(eta_m+1)) - 1
            else:
                delta_q = 1 - (2*(1-u))**(1/(eta_m+1))
            
            delta = (high - low) * delta_q
            mutant[i] += delta
            mutant[i] = np.clip(mutant[i], low, high)
    
    return normalize_weights(mutant)



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


def calculate_crowding_distance(front):
    if not front:
        return
    
    number_of_objectives = len(front[0].fitness)  # Change from objectives to fitness
    for individual in front:
        individual.crowding_distance = 0
    
    for m in range(number_of_objectives):
        front.sort(key=lambda x: x.fitness[m])  # Change from objectives to fitness
        front[0].crowding_distance = front[-1].crowding_distance = float('inf')
        
        if len(front) > 2:
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (front[i + 1].fitness[m] - front[i - 1].fitness[m]) / (front[-1].fitness[m] - front[0].fitness[m])

def sort_population_by_non_domination(population):
    fronts = [[]]
    for individual in population:
        individual.dominated_solutions = set()
        individual.domination_count = 0

        for other in population:
            if individual.dominates(other):
                individual.dominated_solutions.add(other)
            elif other.dominates(individual):
                individual.domination_count += 1
        
        if individual.domination_count == 0:
            individual.rank = 0
            fronts[0].append(individual)
    
    i = 0
    while fronts[i]:
        next_front = []
        for individual in fronts[i]:
            for dominated in individual.dominated_solutions:
                dominated.domination_count -= 1
                if dominated.domination_count == 0:
                    dominated.rank = i + 1
                    next_front.append(dominated)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]

def tournament_selection(ind1, ind2):
    if ind1.rank < ind2.rank:
        return ind1
    elif ind1.rank > ind2.rank:
        return ind2
    else:
        if ind1.crowding_distance > ind2.crowding_distance:
            return ind1
        return ind2


def select_parents(population, num_parents):
    fronts = sort_population_by_non_domination(population)
    for front in fronts:
        calculate_crowding_distance(front)
    
    parents = []
    while len(parents) < num_parents:
        i1, i2 = random.sample(population, 2)
        winner = tournament_selection(i1, i2)
        parents.append(winner)
    
    return parents

def uniform_crossover(parent1_weights, parent2_weights):
    offspring = np.where(np.random.rand(len(parent1_weights)) < 0.5, parent1_weights, parent2_weights)
    offspring /= np.sum(offspring)  # Normalize to ensure weights sum to 1
    return offspring, offspring.copy()  # Return two offsprings for symmetrical crossover

def gaussian_mutation(weights, lower_bound=0, upper_bound=1):
    mutation_rate = ALGORITHM_PARAMS['mutation_rate']
    mutation_shift = ALGORITHM_PARAMS['mutation_shift']
    new_weights = np.copy(weights)
    for i in range(len(weights)):
        if np.random.rand() < mutation_rate:
            mutation = np.random.normal(loc=0, scale=mutation_shift)
            new_weights[i] += mutation

    # Clipping the weights to the bounds before normalization
    new_weights = np.clip(new_weights, lower_bound, upper_bound)

    # Normalize the weights to ensure they sum to 1
    # This step maintains the proportion of each weight relative to the sum of all weights
    new_weights /= np.sum(new_weights)

    return new_weights



def update_pareto_front(population, pareto_front):
    combined = pareto_front + population
    new_front = []

    for candidate in combined:
        dominated = False
        non_dominating = []

        for member in new_front:
            if dominates(candidate, member):
                # If candidate dominates a member, remove the member from the new front
                non_dominating.append(member)
            elif dominates(member, candidate):
                dominated = True
                break

        if not dominated:
            new_front.append(candidate)
            new_front = [x for x in new_front if x not in non_dominating]

    return new_front

def dominates(individual1, individual2):
    better_in_one = False
    for i in range(len(individual1.fitness)):
        if individual1.fitness[i] > individual2.fitness[i]:  # Assuming higher fitness is better for all objectives
            better_in_one = True
        elif individual1.fitness[i] < individual2.fitness[i]:
            return False
    return better_in_one

def elitism_selection(population, n_elites):
    filtered_population = [ind for ind in population if ind.fitness is not None]
    sorted_population = sorted(filtered_population, key=lambda ind: ind.fitness[0], reverse=True)
    return sorted_population[:n_elites]
