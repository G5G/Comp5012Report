# Portfolio Analysis Script

This Python script is designed to parse portfolio data from files and construct covariance matrices based on asset returns and correlations.

## Modules Used

- `numpy`: For mathematical operations and matrix manipulations.

## Functions

### `parse_portfolio_data(file_path)`

Parses data from a given file path to extract:

- Number of assets
- Mean returns per asset
- Standard deviations per asset
- Correlations between assets

#### Parameters:

- `file_path`: Path to the data file.
#### Returns:

- Tuple containing number of assets, list of returns, list of standard deviations, and a dictionary of correlations.

### `construct_covariance_matrix(num_assets, std_devs, correlations)`

Constructs a covariance matrix from standard deviations and correlations.

#### Parameters:

- `num_assets`: Number of assets.
- `std_devs`: List of standard deviations for each asset.
- `correlations`: Dictionary of correlation values between assets.

#### Returns:

- Covariance matrix as a NumPy array.

## Usage Example

```python
files = ["port1.txt", "port2.txt", "port3.txt", "port4.txt", "port5.txt"]
all_data = {}

for file_name in files:
    path = f"data/{file_name}"
    num_assets, returns, std_devs, correlations = parse_portfolio_data(path)
    cov_matrix = construct_covariance_matrix(num_assets, std_devs, correlations)
    all_data[file_name] = {
        "num_assets": num_assets,
        "returns": returns,
        "std_devs": std_devs,
        "cov_matrix": cov_matrix
    }
```
