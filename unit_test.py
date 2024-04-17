import unittest
import numpy as np
from advanced_genetic_operations import polynomial_mutation, gaussian_mutation, sbx_crossover, uniform_crossover
from utils import calculate_expected_return, calculate_portfolio_risk
from config import ALGORITHM_PARAMS
from Individual import Individual

class TestMutationOperators(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.individual = np.array([0.2, 0.3, 0.5])
        self.eta_m = 20  # Example eta for polynomial mutation
    def test_gaussian_mutation(self):

        mutated_individual = gaussian_mutation(
            self.individual,

        )
        self.assertTrue(all(0 <= gene <= 1 for gene in mutated_individual),
                        "All genes should be within [0, 1]")
        self.assertAlmostEqual(np.sum(mutated_individual), 1.0,
                               msg="The weights should sum to 1 after normalization")
        
    def test_polynomial_mutation(self):
        mutated_individual = polynomial_mutation(
            self.individual,
        )

        for gene in mutated_individual:
            self.assertTrue(0 <= gene <= 1, "Gene is out of bounds: {}".format(gene))

        self.assertAlmostEqual(np.sum(mutated_individual), 1.0,
                               msg="The weights should sum to 1 after normalization")


class TestCrossoverOperators(unittest.TestCase):

    def test_sbx_crossover(self):
        np.random.seed(42)
        parent1 = np.array([0.1, 0.2, 0.3, 0.4])
        parent2 = np.array([0.4, 0.3, 0.2, 0.1])
        
        offspring1, offspring2 = sbx_crossover(parent1, parent2)
        
        self.assertFalse(np.array_equal(offspring1, parent1))
        self.assertFalse(np.array_equal(offspring1, parent2))
        self.assertFalse(np.array_equal(offspring2, parent1))
        self.assertFalse(np.array_equal(offspring2, parent2))
        
        self.assertAlmostEqual(np.sum(offspring1), 1.0)
        self.assertAlmostEqual(np.sum(offspring2), 1.0)

    def test_uniform_crossover(self):
        np.random.seed(42)
        parent1 = np.array([0.1, 0.2, 0.3, 0.4])
        parent2 = np.array([0.4, 0.3, 0.2, 0.1])
        
        offspring1, offspring2 = uniform_crossover(parent1, parent2)
        

        self.assertFalse(np.array_equal(offspring1, parent1))
        self.assertFalse(np.array_equal(offspring1, parent2))
        self.assertFalse(np.array_equal(offspring2, parent1))
        self.assertFalse(np.array_equal(offspring2, parent2))
        

        self.assertAlmostEqual(np.sum(offspring1), 1.0)
        self.assertAlmostEqual(np.sum(offspring2), 1.0)





class TestFitnessEvaluation(unittest.TestCase):

    def setUp(self):

        self.returns = np.array([0.1, 0.2, 0.3, 0.4])
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.cov_matrix = np.eye(4)

    def test_calculate_expected_return(self):
        expected_return = calculate_expected_return(self.weights, self.returns)
        self.assertAlmostEqual(expected_return, np.mean(self.returns))

    def test_calculate_portfolio_risk(self):
        portfolio_risk = calculate_portfolio_risk(self.weights, self.cov_matrix)
        expected_risk = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        self.assertAlmostEqual(portfolio_risk, expected_risk)

    def test_individual_fitness(self):
        expected_risk = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        
        individual = Individual(self.weights)
        individual.fitness = [calculate_expected_return(self.weights, self.returns),
                              -calculate_portfolio_risk(self.weights, self.cov_matrix)]
        
        self.assertEqual(individual.fitness[0], np.mean(self.returns))
        self.assertEqual(individual.fitness[1], -expected_risk)



if __name__ == '__main__':
    unittest.main()

