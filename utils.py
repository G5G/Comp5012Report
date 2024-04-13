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
