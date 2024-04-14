import numpy as np

def calculate_expected_return(weights, returns):
    """
    Calculate the expected portfolio return.
    :param weights: np.array of asset weights in the portfolio
    :param returns: np.array of expected returns for each asset
    :return: expected return of the portfolio
    """
    return np.dot(weights, returns)

def calculate_portfolio_risk(weights, cov_matrix):
    """
    Calculate the portfolio risk as the standard deviation of returns.
    :param weights: np.array of asset weights in the portfolio
    :param cov_matrix: covariance matrix of asset returns
    :return: risk (standard deviation) of the portfolio
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def initialize_population(pop_size, num_assets):
    population = np.random.rand(pop_size, num_assets)
    population /= population.sum(axis=1)[:, np.newaxis]  # Normalize to sum to 1
    return population


def evaluate_population(population, all_data):
    fitness = []
    returns = all_data['returns']
    cov_matrix = all_data['cov_matrix']

    for weights in population:
        if len(weights) != len(returns):
            continue  # Skip if dimensions don't match

        portfolio_return = calculate_expected_return(weights, returns)
        portfolio_variance = calculate_portfolio_risk(weights, cov_matrix)

        # Directly append the tuple of return and variance to the fitness list
        fitness.append((portfolio_return, portfolio_variance))

    return fitness

