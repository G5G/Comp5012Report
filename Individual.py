class Individual:
    def __init__(self, weights: object) -> object:
        self.weights = weights
        self.fitness = None  # This holds the objectives
        self.rank = None
        self.crowding_distance = None

    def dominates(self, other):
        better_in_one = False
        for i in range(len(self.fitness)):
            if self.fitness[i] > other.fitness[i]:  # Assuming higher fitness is better for all objectives
                better_in_one = True
            elif self.fitness[i] < other.fitness[i]:
                return False
        return better_in_one
