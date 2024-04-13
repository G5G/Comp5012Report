def uniform_crossover(parent1, parent2):
    """
    Perform uniform crossover between two parents.
    :param parent1: First parent portfolio (numpy array)
    :param parent2: Second parent portfolio (numpy array)
    :return: Offspring portfolio (numpy array)
    """
    mask = np.random.rand(len(parent1)) > 0.5
    offspring = np.where(mask, parent1, parent2)
    offspring /= offspring.sum()  # Normalize to maintain the sum to 1
    return offspring


def mutation(individual, mutation_rate=0.01):
    """
    Mutate an individual by randomly perturbing the asset weights.
    :param individual: Portfolio to mutate (numpy array)
    :param mutation_rate: Rate at which each asset is perturbed
    :return: Mutated portfolio (numpy array)
    """
    perturbation = np.random.normal(loc=0.0, scale=mutation_rate, size=len(individual))
    mutated = individual + perturbation
    mutated = np.clip(mutated, 0, None)  # Ensure no negative weights
    mutated /= mutated.sum()  # Normalize to maintain the sum to 1
    return mutated

