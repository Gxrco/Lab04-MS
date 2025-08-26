"""
Unit tests for genetic algorithm operators.

These tests ensure that all operators produce valid permutations
and maintain the correctness of the genetic algorithm.
"""

import unittest
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.individual import Individual
from core.problem import TSPProblem
from core.crossover import crossover_ox, crossover_pmx
from core.mutation import mutate_swap, mutate_inversion, mutate_scramble
from core.selection import select_tournament
from core.population import Population


class TestOperators(unittest.TestCase):
    """Test cases for GA operators."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple TSP problem
        coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
        self.problem = TSPProblem(coords=coords)
        
        # Create test individuals
        self.individual1 = Individual(tour=[0, 1, 2, 3, 4])
        self.individual2 = Individual(tour=[4, 3, 2, 1, 0])
        self.individual3 = Individual(tour=[2, 0, 4, 1, 3])
        
        # Evaluate fitness
        for ind in [self.individual1, self.individual2, self.individual3]:
            ind.evaluate_fitness(self.problem)
    
    def is_valid_permutation(self, tour, n):
        """Check if tour is a valid permutation of cities 0 to n-1."""
        return (len(tour) == n and 
                set(tour) == set(range(n)) and 
                len(set(tour)) == n)
    
    def test_individual_validity(self):
        """Test that individuals are valid permutations."""
        for ind in [self.individual1, self.individual2, self.individual3]:
            self.assertTrue(ind.is_valid_tour())
            self.assertTrue(self.is_valid_permutation(ind.tour, self.problem.n))
    
    def test_crossover_ox_validity(self):
        """Test that OX crossover produces valid permutations."""
        for _ in range(100):  # Test many times with different random seeds
            child1, child2 = crossover_ox(self.individual1, self.individual2)
            
            # Check that children are valid permutations
            self.assertTrue(child1.is_valid_tour(), 
                           f"Child1 not valid: {child1.tour}")
            self.assertTrue(child2.is_valid_tour(), 
                           f"Child2 not valid: {child2.tour}")
            
            self.assertTrue(self.is_valid_permutation(child1.tour, self.problem.n))
            self.assertTrue(self.is_valid_permutation(child2.tour, self.problem.n))
    
    def test_crossover_pmx_validity(self):
        """Test that PMX crossover produces valid permutations."""
        for _ in range(100):  # Test many times with different random seeds
            child1, child2 = crossover_pmx(self.individual1, self.individual2)
            
            # Check that children are valid permutations
            self.assertTrue(child1.is_valid_tour(), 
                           f"Child1 not valid: {child1.tour}")
            self.assertTrue(child2.is_valid_tour(), 
                           f"Child2 not valid: {child2.tour}")
            
            self.assertTrue(self.is_valid_permutation(child1.tour, self.problem.n))
            self.assertTrue(self.is_valid_permutation(child2.tour, self.problem.n))
    
    def test_crossover_different_parents(self):
        """Test crossover with different parent combinations."""
        parent_pairs = [
            (self.individual1, self.individual2),
            (self.individual1, self.individual3),
            (self.individual2, self.individual3)
        ]
        
        for parent1, parent2 in parent_pairs:
            # Test OX
            child1, child2 = crossover_ox(parent1, parent2)
            self.assertTrue(child1.is_valid_tour())
            self.assertTrue(child2.is_valid_tour())
            
            # Test PMX
            child1, child2 = crossover_pmx(parent1, parent2)
            self.assertTrue(child1.is_valid_tour())
            self.assertTrue(child2.is_valid_tour())
    
    def test_mutate_swap_validity(self):
        """Test that swap mutation produces valid permutations."""
        for _ in range(100):
            for individual in [self.individual1, self.individual2, self.individual3]:
                mutated = mutate_swap(individual)
                
                self.assertTrue(mutated.is_valid_tour(), 
                               f"Mutated not valid: {mutated.tour}")
                self.assertTrue(self.is_valid_permutation(mutated.tour, self.problem.n))
                
                # Should be different from original (with high probability)
                # but not required since swap might select same position twice
    
    def test_mutate_inversion_validity(self):
        """Test that inversion mutation produces valid permutations."""
        for _ in range(100):
            for individual in [self.individual1, self.individual2, self.individual3]:
                mutated = mutate_inversion(individual)
                
                self.assertTrue(mutated.is_valid_tour(), 
                               f"Mutated not valid: {mutated.tour}")
                self.assertTrue(self.is_valid_permutation(mutated.tour, self.problem.n))
    
    def test_mutate_scramble_validity(self):
        """Test that scramble mutation produces valid permutations."""
        for _ in range(100):
            for individual in [self.individual1, self.individual2, self.individual3]:
                mutated = mutate_scramble(individual)
                
                self.assertTrue(mutated.is_valid_tour(), 
                               f"Mutated not valid: {mutated.tour}")
                self.assertTrue(self.is_valid_permutation(mutated.tour, self.problem.n))
    
    def test_mutation_preserves_cities(self):
        """Test that mutations preserve all cities."""
        original_tour = self.individual1.tour.copy()
        
        mutations = [mutate_swap, mutate_inversion, mutate_scramble]
        
        for mutation_func in mutations:
            mutated = mutation_func(self.individual1)
            
            # Should have same cities, just different order
            self.assertEqual(set(original_tour), set(mutated.tour))
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        # Create population
        individuals = [self.individual1, self.individual2, self.individual3]
        population = Population(individuals)
        
        # Test tournament selection multiple times
        for k in [1, 2, 3]:
            selected = select_tournament(population, k)
            
            # Selected individual should be a copy of one from population
            self.assertIn(selected.tour, [ind.tour for ind in individuals])
            self.assertIsInstance(selected, Individual)
    
    def test_fitness_calculation(self):
        """Test fitness calculation."""
        for individual in [self.individual1, self.individual2, self.individual3]:
            # Distance should be positive
            self.assertGreater(individual.distance, 0)
            
            # Fitness should be positive (inverse relationship)
            self.assertGreater(individual.fitness, 0)
            
            # Recalculating should give same result
            old_distance = individual.distance
            individual.evaluate_fitness(self.problem)
            self.assertEqual(old_distance, individual.distance)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with minimum size problem (2 cities)
        coords_small = [(0, 0), (1, 0)]
        problem_small = TSPProblem(coords=coords_small)
        
        ind_small = Individual(tour=[0, 1])
        ind_small.evaluate_fitness(problem_small)
        
        # Test mutations on small problem
        mutated = mutate_swap(ind_small)
        self.assertTrue(mutated.is_valid_tour())
        
        mutated = mutate_inversion(ind_small)
        self.assertTrue(mutated.is_valid_tour())
    
    def test_crossover_preserves_elements(self):
        """Test that crossover preserves all cities."""
        for _ in range(20):
            child1, child2 = crossover_ox(self.individual1, self.individual2)
            
            # Both children should have all cities
            self.assertEqual(set(child1.tour), set(range(self.problem.n)))
            self.assertEqual(set(child2.tour), set(range(self.problem.n)))
            
            child1, child2 = crossover_pmx(self.individual1, self.individual2)
            
            self.assertEqual(set(child1.tour), set(range(self.problem.n)))
            self.assertEqual(set(child2.tour), set(range(self.problem.n)))


class TestProblemClass(unittest.TestCase):
    """Test TSP problem class."""
    
    def test_distance_matrix_symmetry(self):
        """Test that distance matrix is symmetric."""
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        problem = TSPProblem(coords=coords)
        
        n = problem.n
        for i in range(n):
            for j in range(n):
                self.assertEqual(problem.get_distance(i, j), 
                               problem.get_distance(j, i))
    
    def test_distance_matrix_diagonal(self):
        """Test that diagonal distances are zero."""
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        problem = TSPProblem(coords=coords)
        
        for i in range(problem.n):
            self.assertEqual(problem.get_distance(i, i), 0)
    
    def test_tour_distance_calculation(self):
        """Test tour distance calculation."""
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        problem = TSPProblem(coords=coords)
        
        # Simple square: should be 4.0 for tour [0,1,2,3]
        tour = [0, 1, 2, 3]
        distance = problem.calculate_tour_distance(tour)
        
        # Distance should be sum of edges: (0,0)->(1,0) + (1,0)->(1,1) + (1,1)->(0,1) + (0,1)->(0,0)
        # = 1 + 1 + 1 + 1 = 4
        self.assertEqual(distance, 4)


if __name__ == '__main__':
    # Set random seed for reproducible tests
    random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)