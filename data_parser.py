import os
import numpy as np

def parse_portfolio_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    num_assets = int(data[0].strip())  # number of assets
    returns = []
    std_devs = []
    correlations = {}

    for i in range(1, num_assets + 1):
        parts = data[i].strip().split()
        mean_return, std_dev = map(float, parts)
        returns.append(mean_return)
        std_devs.append(std_dev)

    for line in data[num_assets + 1:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        i, j = map(int, parts[:2])
        corr = float(parts[2])
        if i not in correlations:
            correlations[i] = {}
        if j not in correlations:
            correlations[j] = {}
        correlations[i][j] = corr
        correlations[j][i] = corr

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

def read_frontier_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    frontier_points = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            print(f"Skipping line due to incorrect format: '{line}'")
            continue

        mean_return, variance = map(float, parts)
        frontier_points.append((mean_return, variance))

    return frontier_points
