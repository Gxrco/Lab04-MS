"""
Replacement strategies for genetic algorithm.
"""
from typing import List
from .individual import Individual
from .population import Population
from .problem import TSPProblem
from .config import GAConfig
from .selection import select_survivors
from .crossover import create_offspring_crossover
from .mutation import create_offspring_mutation


def create_new_generation(current_population: Population, problem: TSPProblem, 
                         config: GAConfig) -> Population:
    """
    Create new generation using elitism and (μ+λ) replacement strategy.
    
    Args:
        current_population: Current population
        problem: TSP problem instance
        config: GA configuration
        
    Returns:
        New population for next generation
    """
    # Get sizes for each component
    num_survivors = config.get_num_survivors()
    num_crossover = config.get_num_crossover()
    num_mutation = config.get_num_mutation()
    
    # Ensure numbers add up correctly (adjust for rounding errors)
    total_target = config.N
    current_total = num_survivors + num_crossover + num_mutation
    
    if current_total != total_target:
        # Adjust mutation size to match exactly
        num_mutation = total_target - num_survivors - num_crossover
    
    # Select survivors (includes elitism)
    survivors = select_survivors(
        current_population, 
        num_survivors, 
        method=config.selection,
        elitism=config.elitism
    )
    
    # Create offspring through crossover
    crossover_offspring = create_offspring_crossover(
        current_population.individuals,
        num_crossover,
        method=config.crossover
    )
    
    # Create offspring through mutation
    mutation_offspring = create_offspring_mutation(
        current_population.individuals,
        num_mutation,
        method=config.mutation,
        mutation_rate=0.8  # High mutation rate for mutation offspring
    )
    
    # Combine all individuals for new population
    all_individuals = survivors + crossover_offspring + mutation_offspring
    
    # Evaluate fitness for new individuals
    for individual in all_individuals:
        if individual.fitness is None:
            individual.evaluate_fitness(problem)
    
    # Create new population
    new_population = Population(all_individuals[:config.N])  # Ensure exact size
    
    return new_population


def steady_state_replacement(current_population: Population, problem: TSPProblem,
                           config: GAConfig, num_replace: int = 2) -> Population:
    """
    Steady-state replacement: replace a few worst individuals with new offspring.
    
    Args:
        current_population: Current population
        problem: TSP problem instance
        config: GA configuration
        num_replace: Number of individuals to replace
        
    Returns:
        Updated population
    """
    # Sort population by fitness
    current_population.sort_by_fitness()
    
    # Create new offspring
    offspring = []
    
    # Half from crossover, half from mutation
    num_crossover = num_replace // 2
    num_mutation = num_replace - num_crossover
    
    if num_crossover > 0:
        crossover_offspring = create_offspring_crossover(
            current_population.individuals[:config.N//2],  # Use better half as parents
            num_crossover,
            method=config.crossover
        )
        offspring.extend(crossover_offspring)
    
    if num_mutation > 0:
        mutation_offspring = create_offspring_mutation(
            current_population.individuals[:config.N//2],  # Use better half as parents
            num_mutation,
            method=config.mutation,
            mutation_rate=0.7
        )
        offspring.extend(mutation_offspring)
    
    # Evaluate new offspring
    for individual in offspring:
        individual.evaluate_fitness(problem)
    
    # Replace worst individuals with new offspring
    new_individuals = (current_population.individuals[:-num_replace] + 
                      offspring)
    
    new_population = Population(new_individuals)
    new_population.sort_by_fitness()
    
    return new_population


def elitist_replacement(parents: List[Individual], offspring: List[Individual], 
                       population_size: int, elitism_count: int = 1) -> List[Individual]:
    """
    Elitist replacement: ensure best individuals always survive.
    
    Args:
        parents: Parent population
        offspring: Offspring population
        population_size: Target population size
        elitism_count: Number of elite individuals to preserve
        
    Returns:
        New population combining parents and offspring
    """
    # Combine parents and offspring
    all_individuals = parents + offspring
    
    # Sort by fitness (best first)
    all_individuals.sort(key=lambda ind: ind.distance if ind.distance is not None else float('inf'))
    
    # Select best individuals for next generation
    new_population = all_individuals[:population_size]
    
    return new_population


def tournament_replacement(current_population: Population, offspring: List[Individual],
                         tournament_size: int = 3) -> Population:
    """
    Tournament replacement: each offspring competes with random individuals.
    
    Args:
        current_population: Current population
        offspring: New offspring
        tournament_size: Size of tournament for replacement
        
    Returns:
        Updated population
    """
    import random
    
    new_individuals = current_population.individuals.copy()
    
    for child in offspring:
        # Select random individuals for tournament
        competitors_indices = random.sample(range(len(new_individuals)), 
                                          min(tournament_size, len(new_individuals)))
        
        # Find worst competitor
        worst_idx = max(competitors_indices, 
                       key=lambda i: new_individuals[i].distance if new_individuals[i].distance is not None else 0)
        
        # Replace if child is better
        if (child.distance is not None and 
            (new_individuals[worst_idx].distance is None or 
             child.distance < new_individuals[worst_idx].distance)):
            new_individuals[worst_idx] = child
    
    return Population(new_individuals)