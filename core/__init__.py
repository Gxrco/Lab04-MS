"""
Core Genetic Algorithm Module (Person A)

This module contains the core genetic algorithm implementation for TSP.
Includes all GA operators, population management, and the main algorithm.
"""

from .problem import TSPProblem
from .config import GAConfig
from .result import GAResult
from .individual import Individual, create_random_individual, create_greedy_individual
from .population import Population, initialize_population
from .selection import select_tournament, select_rank_based, select_parents, select_survivors
from .crossover import crossover_ox, crossover_pmx, create_offspring_crossover
from .mutation import mutate_swap, mutate_inversion, mutate_scramble, create_offspring_mutation, adaptive_mutate
from .replacement import create_new_generation, steady_state_replacement, elitist_replacement, tournament_replacement
from .diversity import DiversityManager
from .ga_core import run_ga

__version__ = "1.0.0"

__all__ = [
    # Main classes
    'TSPProblem',
    'GAConfig', 
    'GAResult',
    'Individual',
    'Population',
    'DiversityManager',
    
    # Main function
    'run_ga',
    
    # Individual creation
    'create_random_individual',
    'create_greedy_individual',
    
    # Population functions
    'initialize_population',
    
    # Selection operators
    'select_tournament',
    'select_rank_based',
    'select_parents',
    'select_survivors',
    
    # Crossover operators
    'crossover_ox',
    'crossover_pmx',
    'create_offspring_crossover',
    
    # Mutation operators
    'mutate_swap',
    'mutate_inversion',
    'mutate_scramble',
    'create_offspring_mutation',
    'adaptive_mutate',
    
    # Replacement strategies
    'create_new_generation',
    'steady_state_replacement',
    'elitist_replacement',
    'tournament_replacement',
]