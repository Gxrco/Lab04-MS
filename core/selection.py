"""
Selection operators for genetic algorithm.
"""
import random
from typing import List
from .individual import Individual
from .population import Population


def select_tournament(population: Population, k: int = 3) -> Individual:
    """
    Tournament selection: select k random individuals and return the best one.
    
    Args:
        population: Population to select from
        k: Tournament size (number of individuals to compete)
        
    Returns:
        Selected individual (copy)
    """
    # Randomly select k individuals for tournament
    tournament_candidates = random.sample(population.individuals, min(k, len(population.individuals)))
    
    # Return the best individual from tournament (shortest distance)
    winner = min(tournament_candidates, key=lambda ind: ind.distance if ind.distance is not None else float('inf'))
    
    return winner.copy()


def select_rank_based(population: Population) -> Individual:
    """
    Rank-based selection: individuals with better rank have higher probability.
    
    Args:
        population: Population to select from
        
    Returns:
        Selected individual (copy)
    """
    # Sort population by fitness (best first)
    sorted_pop = sorted(population.individuals, key=lambda ind: ind.distance if ind.distance is not None else float('inf'))
    
    # Create rank weights (best individual gets highest weight)
    n = len(sorted_pop)
    weights = [n - i for i in range(n)]  # [n, n-1, n-2, ..., 1]
    
    # Select based on weights
    selected = random.choices(sorted_pop, weights=weights, k=1)[0]
    
    return selected.copy()


def select_parents(population: Population, num_parents: int, method: str = "tournament", 
                  tournament_size: int = 3) -> List[Individual]:
    """
    Select multiple parents for reproduction.
    
    Args:
        population: Population to select from
        num_parents: Number of parents to select
        method: Selection method ("tournament" or "rank")
        tournament_size: Size of tournament (if using tournament selection)
        
    Returns:
        List of selected parents
    """
    parents = []
    
    for _ in range(num_parents):
        if method == "tournament":
            parent = select_tournament(population, tournament_size)
        elif method == "rank":
            parent = select_rank_based(population)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        parents.append(parent)
    
    return parents


def select_survivors(population: Population, num_survivors: int, method: str = "tournament",
                    elitism: int = 0) -> List[Individual]:
    """
    Select survivors for next generation.
    
    Args:
        population: Current population
        num_survivors: Number of survivors to select
        method: Selection method ("tournament" or "rank")
        elitism: Number of best individuals to automatically preserve
        
    Returns:
        List of survivors
    """
    survivors = []
    
    # Add elite individuals first
    if elitism > 0:
        population.sort_by_fitness()
        elite_count = min(elitism, num_survivors, len(population.individuals))
        survivors.extend([ind.copy() for ind in population.individuals[:elite_count]])
    
    # Select remaining survivors
    remaining_slots = num_survivors - len(survivors)
    if remaining_slots > 0:
        additional_survivors = select_parents(population, remaining_slots, method)
        survivors.extend(additional_survivors)
    
    return survivors[:num_survivors]  # Ensure we don't exceed the limit