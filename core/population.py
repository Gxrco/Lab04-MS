"""
Population initialization and management for genetic algorithm.
"""
import random
from typing import List
from .individual import Individual, create_random_individual, create_greedy_individual
from .problem import TSPProblem
from .config import GAConfig


class Population:
    """Manages a population of individuals."""
    
    def __init__(self, individuals: List[Individual]):
        """Initialize population with list of individuals."""
        self.individuals = individuals
        self.size = len(individuals)
    
    def evaluate_all(self, problem: TSPProblem):
        """Evaluate fitness for all individuals in population."""
        for individual in self.individuals:
            individual.evaluate_fitness(problem)
    
    def sort_by_fitness(self):
        """Sort population by fitness (best first - shortest distance)."""
        self.individuals.sort(key=lambda ind: ind.distance if ind.distance is not None else float('inf'))
    
    def get_best(self) -> Individual:
        """Get best individual (shortest distance)."""
        return min(self.individuals, key=lambda ind: ind.distance if ind.distance is not None else float('inf'))
    
    def get_worst(self) -> Individual:
        """Get worst individual (longest distance)."""
        return max(self.individuals, key=lambda ind: ind.distance if ind.distance is not None else float('inf'))
    
    def get_average_distance(self) -> float:
        """Get average distance of population."""
        valid_distances = [ind.distance for ind in self.individuals if ind.distance is not None]
        return sum(valid_distances) / len(valid_distances) if valid_distances else float('inf')
    
    def calculate_diversity(self) -> float:
        """Calculate diversity as percentage of unique individuals."""
        unique_tours = set(tuple(ind.tour) for ind in self.individuals)
        return len(unique_tours) / self.size if self.size > 0 else 0.0
    
    def remove_duplicates(self, problem: TSPProblem):
        """Remove duplicate individuals and replace with random ones."""
        seen_tours = set()
        new_individuals = []
        
        for ind in self.individuals:
            tour_tuple = tuple(ind.tour)
            if tour_tuple not in seen_tours:
                seen_tours.add(tour_tuple)
                new_individuals.append(ind)
            else:
                # Replace duplicate with random individual
                new_ind = create_random_individual(problem.n)
                new_ind.evaluate_fitness(problem)
                new_individuals.append(new_ind)
        
        self.individuals = new_individuals
    
    def __len__(self) -> int:
        """Get population size."""
        return self.size
    
    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]
    
    def __setitem__(self, index: int, individual: Individual):
        """Set individual by index."""
        self.individuals[index] = individual


def initialize_population(problem: TSPProblem, config: GAConfig) -> Population:
    """
    Initialize population with random individuals and optional greedy seeding.
    
    Args:
        problem: TSP problem instance
        config: GA configuration
        
    Returns:
        Initialized population
    """
    individuals = []
    
    # Create mostly random individuals
    num_random = max(1, config.N - config.N // 10)  # 90% random
    for _ in range(num_random):
        individuals.append(create_random_individual(problem.n))
    
    # Add some greedy individuals for diversity (10% or at least 1)
    num_greedy = config.N - num_random
    for i in range(num_greedy):
        start_city = i % problem.n  # Different starting cities
        individuals.append(create_greedy_individual(problem, start_city))
    
    # Ensure we have exactly N individuals
    while len(individuals) < config.N:
        individuals.append(create_random_individual(problem.n))
    
    # Truncate if we have too many (shouldn't happen)
    individuals = individuals[:config.N]
    
    population = Population(individuals)
    population.evaluate_all(problem)
    
    return population