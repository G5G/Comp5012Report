DATA_FILES_PATH = "data"
PORTFOLIO_PREFIX = 'port'
FRONTIER_PREFIX = 'portef'
NO_OF_FILES = 1
NO_OF_TRIALS = 50
# Define algorithm parameters
ALGORITHM_PARAMS = {
    "POP_SIZE": 250,
    "NUM_GENERATIONS": 250,
    "CROSSOVER_RATE" : 0.5,
    'mutation_rate': 0.4,
    'mutation_shift': 0.02,
    "ELITISM_COUNT": 20,  # Number of elites to retain each generation, CONSIDERING POP SIZE

    "crossover_type": "sbx_crossover",  # Options: 'uniform', 'sbx'
    "mutation_type": "gaussian_mutation", # Options: 'gaussian', 'polynomial'

    # adjust if sbx is choosen for crossover
    "SBX_ETA": 15, 
    # adjust if polynomical is choosen for mutation
    "POLY_ETA": 20,  

}
