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
