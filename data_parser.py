import os

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

#todo: not sure where the frontier to use
def read_frontier(directory_path, prefix):
    frontier_points_list = []

    for filename in os.listdir(directory_path):
        if filename.startswith(prefix):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            frontier_points = []
            for line in lines:
                mean_return, variance = map(float, line.split())
                frontier_points.append((mean_return, variance))

            frontier_points_list.append(frontier_points)

    return frontier_points_list
# frontier_points_list = read_frontier(DATA_FILES_PATH, FRONTIER_PREFIX)

#print(all_data)
#print("_____")