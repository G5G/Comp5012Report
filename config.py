DATA_FILES_PATH = "data"
PORTFOLIO_PREFIX = 'port'
FRONTIER_PREFIX = 'portef'
NO_OF_FILES = 1
NO_OF_TRIALS = 50
# Define algorithm parameters
ALGORITHM_PARAMS = {
    "POP_SIZE": 200,
    "NUM_GENERATIONS": 250,
    "CROSSOVER_RATE" : 0.3,
    'mutation_rate': 0.05,
    'mutation_shift': 0.1,
    "ELITISM_COUNT": 5,  # Number of elites to retain each generation, CONSIDERING POP SIZE

    "crossover_type": "sbx_crossover",  # Options: 'uniform_crossover', 'sbx_crossover'
    "mutation_type": "gaussian_mutation", # Options: 'gaussian_mutation', 'polynomial_mutation'

    # adjust if sbx is choosen for crossover
    "SBX_ETA": 15, 
    # adjust if polynomical is choosen for mutation
    "POLY_ETA": 20,  

}

