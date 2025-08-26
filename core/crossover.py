"""
Crossover operators for genetic algorithm (TSP).
"""
import random
from typing import Tuple, List
from .individual import Individual


def crossover_ox(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Order Crossover (OX) for TSP.
    
    Algorithm:
    1. Select two random cut points
    2. Copy segment between cut points from parent1 to child1
    3. Fill remaining positions with cities from parent2 in order they appear
    4. Repeat for child2 (swap parents)
    
    Args:
        parent1: First parent
        parent2: Second parent
        
    Returns:
        Tuple of two children
    """
    n = len(parent1.tour)
    
    # Select two random cut points
    cut1 = random.randint(0, n - 1)
    cut2 = random.randint(0, n - 1)
    
    # Ensure cut1 <= cut2
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    
    # Create child1
    child1_tour = [-1] * n
    child2_tour = [-1] * n
    
    # Copy segment from parent1 to child1
    child1_tour[cut1:cut2+1] = parent1.tour[cut1:cut2+1]
    
    # Copy segment from parent2 to child2
    child2_tour[cut1:cut2+1] = parent2.tour[cut1:cut2+1]
    
    # Fill remaining positions for child1
    _fill_remaining_ox(child1_tour, parent2.tour, cut1, cut2)
    
    # Fill remaining positions for child2
    _fill_remaining_ox(child2_tour, parent1.tour, cut1, cut2)
    
    child1 = Individual(tour=child1_tour)
    child2 = Individual(tour=child2_tour)
    
    return child1, child2


def _fill_remaining_ox(child_tour: List[int], parent_tour: List[int], cut1: int, cut2: int):
    """Helper function to fill remaining positions in OX crossover."""
    n = len(child_tour)
    used_cities = set(child_tour[cut1:cut2+1])
    
    # Get unused cities from parent in order
    unused_cities = []
    for city in parent_tour:
        if city not in used_cities:
            unused_cities.append(city)
    
    # Fill positions after cut2
    pos = (cut2 + 1) % n
    city_idx = 0
    
    while city_idx < len(unused_cities):
        if child_tour[pos] == -1:  # Empty position
            child_tour[pos] = unused_cities[city_idx]
            city_idx += 1
        pos = (pos + 1) % n


def crossover_pmx(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Partially Mapped Crossover (PMX) for TSP.
    
    Algorithm:
    1. Select two random cut points
    2. Copy segment between cut points from each parent
    3. Create mapping from the exchanged segments
    4. Fill remaining positions using the mapping to avoid duplicates
    
    Args:
        parent1: First parent
        parent2: Second parent
        
    Returns:
        Tuple of two children
    """
    n = len(parent1.tour)
    
    # Select two random cut points
    cut1 = random.randint(0, n - 1)
    cut2 = random.randint(0, n - 1)
    
    # Ensure cut1 <= cut2
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    
    # Initialize children with -1 (empty)
    child1_tour = [-1] * n
    child2_tour = [-1] * n
    
    # Copy segments from opposite parents
    child1_tour[cut1:cut2+1] = parent2.tour[cut1:cut2+1]
    child2_tour[cut1:cut2+1] = parent1.tour[cut1:cut2+1]
    
    # Create mapping for PMX
    mapping1 = {}
    mapping2 = {}
    
    for i in range(cut1, cut2 + 1):
        if parent1.tour[i] != parent2.tour[i]:  # Only map if different
            mapping1[parent1.tour[i]] = parent2.tour[i]
            mapping2[parent2.tour[i]] = parent1.tour[i]
    
    # Fill remaining positions for child1
    _fill_pmx_positions(child1_tour, parent1.tour, mapping1, cut1, cut2)
    
    # Fill remaining positions for child2
    _fill_pmx_positions(child2_tour, parent2.tour, mapping2, cut1, cut2)
    
    child1 = Individual(tour=child1_tour)
    child2 = Individual(tour=child2_tour)
    
    return child1, child2


def _fill_pmx_positions(child_tour: List[int], parent_tour: List[int], 
                       mapping: dict, cut1: int, cut2: int):
    """Fill remaining positions in PMX crossover."""
    n = len(child_tour)
    
    for i in range(n):
        if child_tour[i] == -1:  # Empty position
            city = parent_tour[i]
            
            # Check if city is already in child (in copied segment)
            while city in child_tour:
                if city in mapping:
                    city = mapping[city]
                else:
                    # Find an unmapped city
                    for candidate in range(n):
                        if candidate not in child_tour:
                            city = candidate
                            break
                    break
            
            child_tour[i] = city


def create_offspring_crossover(parents: List[Individual], num_offspring: int, 
                              method: str = "OX") -> List[Individual]:
    """
    Create offspring using crossover.
    
    Args:
        parents: List of parent individuals
        num_offspring: Number of offspring to create
        method: Crossover method ("OX" or "PMX")
        
    Returns:
        List of offspring
    """
    offspring = []
    
    while len(offspring) < num_offspring:
        # Select two random parents
        parent1, parent2 = random.sample(parents, 2)
        
        if method == "OX":
            child1, child2 = crossover_ox(parent1, parent2)
        elif method == "PMX":
            child1, child2 = crossover_pmx(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {method}")
        
        offspring.extend([child1, child2])
    
    return offspring[:num_offspring]  # Return exactly num_offspring children