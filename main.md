# Portfolio Population Initialization Function

This module includes a function to initialize a population of portfolios with random weights that sum to one. This is particularly useful for simulations or optimization processes in financial modeling, such as genetic algorithms for portfolio optimization.

## Function

### initialize_population

Initializes a population of portfolios with random weights.

**Arguments:**

- `pop_size`: `int` - Number of portfolios in the population.
- `num_assets`: `int` - Number of assets in each portfolio.

**Returns:**

- `numpy.array` - A numpy array representing the initial population, where each row corresponds to a portfolio and the values in each row sum to 1.

## Dependencies

- NumPy: This function utilizes NumPy for generating random numbers and performing array operations. Ensure that NumPy is installed and imported as `np`.

## Notes

- The function generates random numbers for each asset in each portfolio, then normalizes these numbers so that each portfolio's weights sum to 1.
- This normalization ensures that each portfolio's asset allocation is proportionally scaled and suitable for portfolio optimization models.
