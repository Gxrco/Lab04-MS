"""
Mutation operators for genetic algorithm (TSP).
"""
import random
from typing import List
from .individual import Individual


def mutate_swap(individual: Individual) -> Individual:
    """
    Swap mutation: randomly swap two cities in the tour.
    
    Args:
        individual: Individual to mutate
        
    Returns:
        Mutated individual (new copy)
    """
    mutated = individual.copy()
    n = len(mutated.tour)
    
    if n < 2:
        return mutated
    
    # Select two random positions
    pos1 = random.randint(0, n - 1)
    pos2 = random.randint(0, n - 1)
    
    # Ensure positions are different
    while pos1 == pos2:
        pos2 = random.randint(0, n - 1)
    
    # Swap cities at selected positions
    mutated.tour[pos1], mutated.tour[pos2] = mutated.tour[pos2], mutated.tour[pos1]
    
    # Reset fitness (needs to be recalculated)
    mutated.fitness = None
    mutated.distance = None
    
    return mutated


def mutate_inversion(individual: Individual) -> Individual:
    """
    Inversion mutation (2-opt): reverse the order of cities between two points.
    
    This is a local search move that's particularly effective for TSP.
    
    Args:
        individual: Individual to mutate
        
    Returns:
        Mutated individual (new copy)
    """
    mutated = individual.copy()
    n = len(mutated.tour)
    
    if n < 2:
        return mutated
    
    # Select two random cut points
    cut1 = random.randint(0, n - 1)
    cut2 = random.randint(0, n - 1)
    
    # Ensure cut1 <= cut2 and they're different
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    
    if cut1 == cut2:
        # If same position, just do a small inversion
        if cut1 < n - 1:
            cut2 = cut1 + 1
        else:
            cut1 = cut1 - 1
    
    # Reverse the segment between cut1 and cut2 (inclusive)
    mutated.tour[cut1:cut2+1] = reversed(mutated.tour[cut1:cut2+1])
    
    # Reset fitness (needs to be recalculated)
    mutated.fitness = None
    mutated.distance = None
    
    return mutated


def mutate_scramble(individual: Individual) -> Individual:
    """
    Scramble mutation: randomly shuffle cities in a randomly selected segment.
    
    Args:
        individual: Individual to mutate
        
    Returns:
        Mutated individual (new copy)
    """
    mutated = individual.copy()
    n = len(mutated.tour)
    
    if n < 2:
        return mutated
    
    # Select segment to scramble
    cut1 = random.randint(0, n - 1)
    cut2 = random.randint(0, n - 1)
    
    # Ensure cut1 <= cut2
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    
    # Extract segment and shuffle it
    segment = mutated.tour[cut1:cut2+1]
    random.shuffle(segment)
    
    # Replace segment in tour
    mutated.tour[cut1:cut2+1] = segment
    
    # Reset fitness (needs to be recalculated)
    mutated.fitness = None
    mutated.distance = None
    
    return mutated


def create_offspring_mutation(parents: List[Individual], num_offspring: int,
                             method: str = "inversion", mutation_rate: float = 0.1) -> List[Individual]:
    """
    Create offspring using mutation.
    
    Args:
        parents: List of parent individuals
        num_offspring: Number of offspring to create
        method: Mutation method ("swap", "inversion", or "scramble")
        mutation_rate: Probability of actually applying mutation
        
    Returns:
        List of offspring
    """
    offspring = []
    
    for _ in range(num_offspring):
        # Select random parent
        parent = random.choice(parents)
        
        # Apply mutation with given probability
        if random.random() < mutation_rate:
            if method == "swap":
                child = mutate_swap(parent)
            elif method == "inversion":
                child = mutate_inversion(parent)
            elif method == "scramble":
                child = mutate_scramble(parent)
            else:
                raise ValueError(f"Unknown mutation method: {method}")
        else:
            # No mutation, just copy parent
            child = parent.copy()
        
        offspring.append(child)
    
    return offspring


def adaptive_mutate(individual: Individual, stagnation_generations: int,
                   base_method: str = "inversion") -> Individual:
    """
    Apply adaptive mutation based on stagnation.
    Higher stagnation leads to more aggressive mutation.
    
    Args:
        individual: Individual to mutate
        stagnation_generations: Number of generations without improvement
        base_method: Base mutation method to use
        
    Returns:
        Mutated individual
    """
    # Increase mutation aggressiveness with stagnation
    if stagnation_generations < 10:
        # Normal mutation
        if base_method == "swap":
            return mutate_swap(individual)
        else:
            return mutate_inversion(individual)
    
    elif stagnation_generations < 25:
        # More aggressive: multiple mutations
        mutated = individual.copy()
        num_mutations = min(3, stagnation_generations // 5)
        
        for _ in range(num_mutations):
            if base_method == "swap":
                mutated = mutate_swap(mutated)
            else:
                mutated = mutate_inversion(mutated)
        
        return mutated
    
    else:
        # Very aggressive: scramble large segments
        return mutate_scramble(individual)