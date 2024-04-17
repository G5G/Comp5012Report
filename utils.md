# Portfolio Analysis Functions

This module contains functions to calculate the expected return and risk of an investment portfolio based on asset weights and returns.

## Functions

### calculate_expected_return

Calculates the expected return of a portfolio.

**Arguments:**

- `weights`: `np.array` - Array of asset weights in the portfolio.
- `returns`: `np.array` - Array of expected returns for each asset.

**Returns:**

- `float` - The expected return of the portfolio.

### calculate_portfolio_risk

Calculates the portfolio risk as the standard deviation of returns.

**Arguments:**

- `weights`: `np.array` - Array of asset weights in the portfolio.
- `cov_matrix`: `np.array` - Covariance matrix of asset returns.

**Returns:**

- `float` - The risk (standard deviation) of the portfolio as a float.

## Dependencies

- NumPy: This module uses NumPy for matrix operations and mathematical computations. Ensure that NumPy is installed and imported as `np`.

## Notes

- The `calculate_expected_return` function uses the dot product to compute the return, which assumes that both weights and returns are properly aligned.
- The `calculate_portfolio_risk` function computes the portfolio variance first by matrix multiplication of weights with the covariance matrix and then calculates the standard deviation.
