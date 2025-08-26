"""
Diversity control and adaptive mechanisms for genetic algorithm.
"""
import random
from typing import List, Set, Tuple
from .individual import Individual
from .population import Population
from .problem import TSPProblem
from .mutation import adaptive_mutate


class DiversityManager:
    """Manages population diversity and adaptive mechanisms."""
    
    def __init__(self, min_diversity_threshold: float = 0.3, 
                 stagnation_threshold: int = 20):
        """
        Initialize diversity manager.
        
        Args:
            min_diversity_threshold: Minimum diversity ratio to maintain
            stagnation_threshold: Generations without improvement before adaptation
        """
        self.min_diversity_threshold = min_diversity_threshold
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.last_best_fitness = float('inf')
    
    def update_stagnation_counter(self, current_best_fitness: float):
        """Update stagnation counter based on fitness improvement."""
        if current_best_fitness < self.last_best_fitness:
            # Improvement found, reset counter
            self.stagnation_counter = 0
            self.last_best_fitness = current_best_fitness
        else:
            # No improvement
            self.stagnation_counter += 1
        
        self.best_fitness_history.append(current_best_fitness)
    
    def is_stagnated(self) -> bool:
        """Check if algorithm is stagnated."""
        return self.stagnation_counter >= self.stagnation_threshold
    
    def get_stagnation_level(self) -> int:
        """Get current stagnation level."""
        return self.stagnation_counter
    
    def calculate_diversity_metrics(self, population: Population) -> dict:
        """
        Calculate various diversity metrics.
        
        Args:
            population: Population to analyze
            
        Returns:
            Dictionary with diversity metrics
        """
        n = len(population.individuals)
        if n == 0:
            return {'unique_ratio': 0.0, 'hamming_diversity': 0.0, 'fitness_variance': 0.0}
        
        # Unique individuals ratio
        unique_tours = set(tuple(ind.tour) for ind in population.individuals)
        unique_ratio = len(unique_tours) / n
        
        # Average Hamming distance between tours
        hamming_diversity = self._calculate_hamming_diversity(population.individuals)
        
        # Fitness variance
        fitness_values = [ind.distance for ind in population.individuals if ind.distance is not None]
        if len(fitness_values) > 1:
            mean_fitness = sum(fitness_values) / len(fitness_values)
            fitness_variance = sum((f - mean_fitness) ** 2 for f in fitness_values) / len(fitness_values)
        else:
            fitness_variance = 0.0
        
        return {
            'unique_ratio': unique_ratio,
            'hamming_diversity': hamming_diversity,
            'fitness_variance': fitness_variance,
            'unique_count': len(unique_tours)
        }
    
    def _calculate_hamming_diversity(self, individuals: List[Individual]) -> float:
        """Calculate average Hamming distance between all pairs of tours."""
        if len(individuals) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        tour_length = len(individuals[0].tour)
        
        for i in range(len(individuals)):
            for j in range(i + 1, len(individuals)):
                # Calculate Hamming distance (number of different positions)
                hamming_dist = sum(1 for k in range(tour_length) 
                                 if individuals[i].tour[k] != individuals[j].tour[k])
                total_distance += hamming_dist / tour_length  # Normalize by tour length
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def maintain_diversity(self, population: Population, problem: TSPProblem) -> Population:
        """
        Maintain population diversity by removing duplicates and adding variation.
        
        Args:
            population: Current population
            problem: TSP problem instance
            
        Returns:
            Population with maintained diversity
        """
        diversity_metrics = self.calculate_diversity_metrics(population)
        
        if diversity_metrics['unique_ratio'] < self.min_diversity_threshold:
            # Diversity is too low, take action
            population = self._increase_diversity(population, problem)
        
        return population
    
    def _increase_diversity(self, population: Population, problem: TSPProblem) -> Population:
        """Increase population diversity."""
        # Remove exact duplicates and replace with mutated versions
        seen_tours = set()
        new_individuals = []
        
        # Sort by fitness to keep best individuals
        population.sort_by_fitness()
        
        for ind in population.individuals:
            tour_tuple = tuple(ind.tour)
            if tour_tuple not in seen_tours:
                seen_tours.add(tour_tuple)
                new_individuals.append(ind)
            else:
                # Replace duplicate with heavily mutated version
                mutated = self._heavy_mutate(ind)
                mutated.evaluate_fitness(problem)
                new_individuals.append(mutated)
        
        return Population(new_individuals)
    
    def _heavy_mutate(self, individual: Individual) -> Individual:
        """Apply heavy mutation to increase diversity."""
        mutated = individual.copy()
        
        # Apply multiple mutations
        num_mutations = random.randint(2, 5)
        for _ in range(num_mutations):
            if random.random() < 0.5:
                # Swap mutation
                n = len(mutated.tour)
                if n >= 2:
                    pos1, pos2 = random.sample(range(n), 2)
                    mutated.tour[pos1], mutated.tour[pos2] = mutated.tour[pos2], mutated.tour[pos1]
            else:
                # Inversion mutation
                n = len(mutated.tour)
                if n >= 2:
                    cut1, cut2 = sorted(random.sample(range(n), 2))
                    mutated.tour[cut1:cut2+1] = reversed(mutated.tour[cut1:cut2+1])
        
        # Reset fitness
        mutated.fitness = None
        mutated.distance = None
        
        return mutated
    
    def apply_adaptive_mechanisms(self, population: Population, problem: TSPProblem) -> Population:
        """
        Apply adaptive mechanisms based on stagnation level.
        
        Args:
            population: Current population
            problem: TSP problem instance
            
        Returns:
            Population with adaptive modifications applied
        """
        if not self.is_stagnated():
            return population
        
        stagnation_level = self.get_stagnation_level()
        
        if stagnation_level < 50:
            # Moderate stagnation: increase mutation in worst individuals
            return self._moderate_adaptation(population, problem)
        else:
            # High stagnation: restart part of population
            return self._aggressive_adaptation(population, problem)
    
    def _moderate_adaptation(self, population: Population, problem: TSPProblem) -> Population:
        """Apply moderate adaptive changes."""
        population.sort_by_fitness()
        
        # Mutate worst 30% of population
        split_point = int(len(population.individuals) * 0.7)
        good_individuals = population.individuals[:split_point]
        bad_individuals = population.individuals[split_point:]
        
        # Apply adaptive mutation to bad individuals
        mutated_individuals = []
        for ind in bad_individuals:
            mutated = adaptive_mutate(ind, self.stagnation_counter)
            mutated.evaluate_fitness(problem)
            mutated_individuals.append(mutated)
        
        new_individuals = good_individuals + mutated_individuals
        return Population(new_individuals)
    
    def _aggressive_adaptation(self, population: Population, problem: TSPProblem) -> Population:
        """Apply aggressive adaptive changes (partial restart)."""
        population.sort_by_fitness()
        
        # Keep only best 20% of population
        keep_count = max(1, int(len(population.individuals) * 0.2))
        elite_individuals = population.individuals[:keep_count]
        
        # Generate new random individuals for the rest
        new_individuals = list(elite_individuals)
        
        while len(new_individuals) < len(population.individuals):
            # Create new individual
            new_ind = Individual(n=problem.n)
            new_ind.evaluate_fitness(problem)
            new_individuals.append(new_ind)
        
        # Reset stagnation counter after restart
        self.stagnation_counter = 0
        
        return Population(new_individuals)