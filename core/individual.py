"""
Individual representation and fitness functions for TSP genetic algorithm.
"""
import random
from typing import Optional
from .problem import TSPProblem


class Individual:
    """Represents an individual in the genetic algorithm (TSP tour)."""
    
    def __init__(self, tour: Optional[list[int]] = None, n: Optional[int] = None):
        """
        Initialize individual.
        
        Args:
            tour: Tour as permutation of city indices (0 to n-1)
            n: Number of cities (used for random initialization if tour not provided)
        """
        if tour is not None:
            self.tour = tour.copy()
            self.n = len(tour)
        elif n is not None:
            self.tour = list(range(n))
            random.shuffle(self.tour)
            self.n = n
        else:
            raise ValueError("Either tour or n must be provided")
        
        self.fitness: Optional[float | int] = None
        self.distance: Optional[float | int] = None
    
    def evaluate_fitness(self, problem: TSPProblem):
        """Evaluate fitness (tour distance) using problem instance."""
        self.distance = problem.calculate_tour_distance(self.tour)
        # Fitness is inverse of distance (higher fitness = shorter distance)
        self.fitness = 1.0 / (1.0 + self.distance)
    
    def is_valid_tour(self) -> bool:
        """Check if tour is a valid permutation."""
        return (len(self.tour) == self.n and 
                set(self.tour) == set(range(self.n)) and
                len(set(self.tour)) == self.n)
    
    def copy(self) -> 'Individual':
        """Create a deep copy of the individual."""
        new_individual = Individual(tour=self.tour)
        new_individual.fitness = self.fitness
        new_individual.distance = self.distance
        return new_individual
    
    def __lt__(self, other: 'Individual') -> bool:
        """Compare individuals by distance (for sorting)."""
        if self.distance is None or other.distance is None:
            return False
        return self.distance < other.distance
    
    def __eq__(self, other: 'Individual') -> bool:
        """Check if two individuals have the same tour."""
        return self.tour == other.tour
    
    def __hash__(self) -> int:
        """Hash based on tour for set operations."""
        return hash(tuple(self.tour))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Individual(tour={self.tour}, distance={self.distance})"


def create_random_individual(n: int) -> Individual:
    """Create a random individual (random tour)."""
    return Individual(n=n)


def create_greedy_individual(problem: TSPProblem, start_city: int = 0) -> Individual:
    """Create greedy nearest neighbor individual."""
    n = problem.n
    tour = [start_city]
    remaining = set(range(n)) - {start_city}
    
    current_city = start_city
    while remaining:
        # Find nearest unvisited city
        nearest_city = min(remaining, key=lambda city: problem.get_distance(current_city, city))
        tour.append(nearest_city)
        remaining.remove(nearest_city)
        current_city = nearest_city
    
    individual = Individual(tour=tour)
    individual.evaluate_fitness(problem)
    return individual