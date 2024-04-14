import numpy as np

def normalize_weights(weights):
    return weights / np.sum(weights)

def sbx_crossover(parent1, parent2, eta_c=15, low=0, high=1):
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

def polynomial_mutation(individual, mutation_rate, eta_m=20, low=0, high=1):
    mutant = np.copy(individual)
    for i in range(len(mutant)):
        if np.random.rand() < mutation_rate:
            u = np.random.rand()
            delta = (2*u)**(1/(eta_m+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(eta_m+1))
            
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

def pareto_rank_and_crowding(population, fitness):
    ranks = {}
    crowding_distances = {i: 0 for i in range(len(fitness))}  # Initialize crowding distances

    # Example logic to assign ranks and compute crowding distances
    # This is a placeholder for actual Pareto ranking and crowding distance calculation
    for i, (r1, v1) in enumerate(fitness):
        ranks[i] = sum(r2 >= r1 and v2 <= v1 for j, (r2, v2) in enumerate(fitness) if i != j)
    
    # Sort based on rank and then by crowding distance (larger is better)
    sorted_indices = sorted(range(len(fitness)), key=lambda i: (ranks[i], -crowding_distances[i]))

    return [population[i] for i in sorted_indices], [fitness[i] for i in sorted_indices]

def integrate_elitism(population, fitness, elite_size=10):
    sorted_population, sorted_fitness = pareto_rank_and_crowding(population, fitness)
    elite_individuals = sorted_population[:elite_size]
    return elite_individuals
