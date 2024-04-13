import numpy as np
def parse_portfolio_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    num_assets = int(data[0].strip())  # number of assets
    returns = []
    std_devs = []
    correlations = {}

    # Read mean returns and standard deviations
    for i in range(1, num_assets + 1):
        parts = data[i].strip().split()
        mean_return, std_dev = map(float, parts)
        returns.append(mean_return)
        std_devs.append(std_dev)
    
    # Read correlations and ensure symmetry
    for line in data[num_assets + 1:]:
        parts = line.strip().split()
        if len(parts) < 3:
            print(line)
            continue  # Skip lines that do not have enough data
        i, j = map(int, parts[:2])
        corr = float(parts[2])
        if i not in correlations:
            correlations[i] = {}
        if j not in correlations:
            correlations[j] = {}
        correlations[i][j] = corr
        correlations[j][i] = corr  # Ensure the matrix is symmetric

    return num_assets, returns, std_devs, correlations

def construct_covariance_matrix(num_assets, std_devs, correlations):
    cov_matrix = np.zeros((num_assets, num_assets))
    for i in range(1, num_assets + 1):
        for j in range(1, num_assets + 1):
            if i == j:
                cov_matrix[i-1][j-1] = std_devs[i-1]**2
            else:
                cov_matrix[i-1][j-1] = correlations[i][j] * std_devs[i-1] * std_devs[j-1]
    return cov_matrix

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

#print(all_data)
#print("_____")