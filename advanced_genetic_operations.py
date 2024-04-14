import numpy as np

def sbx_crossover(parent1, parent2, eta_c=15, low=0, high=1):
    """
    Perform Simulated Binary Crossover between two parents.
    :param parent1: First parent array
    :param parent2: Second parent array
    :param eta_c: Crossover distribution index (larger values make offspring similar to parents)
    :param low: Lower bound of the values
    :param high: Upper bound of the values
    :return: Two offspring resulting from the crossover
    """
    rand = np.random.rand(parent1.shape[0])
    gamma = np.empty(parent1.shape[0])
    mask = rand <= 0.5
    
    gamma[mask] = (2 * rand[mask]) ** (1 / (eta_c + 1))
    gamma[~mask] = (1 / (2 * (1 - rand[~mask]))) ** (1 / (eta_c + 1))
    
    offspring1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
    offspring2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)
    
    offspring1 = np.clip(offspring1, low, high)
    offspring2 = np.clip(offspring2, low, high)
    
    return offspring1, offspring2

def polynomial_mutation(individual, mutation_rate, eta_m=20, low=0, high=1):
    """
    Mutate an individual using Polynomial Mutation.
    :param individual: Individual array to mutate
    :param mutation_rate: Probability of mutation
    :param eta_m: Mutation distribution index (higher values make mutants closer to the original)
    :param low: Lower bound of the values
    :param high: Upper bound of the values
    :return: Mutated individual
    """
    mutant = np.copy(individual)
    for i in range(len(mutant)):
        if np.random.rand() < mutation_rate:
            u = np.random.rand()
            delta = 0
            if u < 0.5:
                delta = (2*u)**(1/(eta_m+1)) - 1
            else:
                delta = 1 - (2*(1-u))**(1/(eta_m+1))
                
            mutant[i] += delta
            mutant[i] = np.clip(mutant[i], low, high)
    
    return mutant

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
    # Sort the population based on fitness; smaller fitness values are better
    sorted_population = sorted(zip(population, fitness), key=lambda x: x[1])
    elite_individuals = [ind for ind, fit in sorted_population[:elite_size]]
    return elite_individuals