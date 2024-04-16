DATA_FILES_PATH = "data"
PORTFOLIO_PREFIX = 'port'
FRONTIER_PREFIX = 'portef'
NO_OF_FILES = 1
# Define algorithm parameters
ALGORITHM_PARAMS = {
    "POP_SIZE": 100,
    "NUM_GENERATIONS": 50,
    "CROSSOVER_RATE" : 0.2,
    'mutation_rate': 0.2,
    'mutation_shift': 0.05,
    "ELITISM_COUNT": 3,  # Number of elites to retain each generation, CONSIDERING POP SIZE

    "crossover_type": "sbx",  # Options: 'uniform', 'sbx'
    "mutation_type": "polynomial", # Options: 'uniform', 'polynomial'

    # adjust if sbx is choosen for crossover
    "SBX_ETA": 15, 
    # adjust if polynomical is choosen for mutation
    "POLY_ETA": 20,  

}
