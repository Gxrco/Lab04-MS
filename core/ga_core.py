"""
Main genetic algorithm core module.
"""
import random
import time
from typing import Optional, List, Callable, Any

from .problem import TSPProblem
from .config import GAConfig
from .result import GAResult
from .population import initialize_population
from .replacement import create_new_generation
from .diversity import DiversityManager


def run_ga(problem: TSPProblem, config: GAConfig, 
          callbacks: Optional[List[Callable]] = None) -> GAResult:
    """
    Run genetic algorithm for TSP.
    
    Args:
        problem: TSP problem instance
        config: GA configuration
        callbacks: Optional list of callback functions called each iteration
        
    Returns:
        GA results with best solution and history
    """
    # Set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
    
    # Initialize result object
    result = GAResult()
    
    # Initialize diversity manager
    diversity_manager = DiversityManager(
        min_diversity_threshold=0.3,
        stagnation_threshold=20
    )
    
    # Initialize population
    print(f"Initializing population of size {config.N}...")
    population = initialize_population(problem, config)
    
    # Track best solution
    best_individual = population.get_best()
    result.update_best(best_individual.tour, best_individual.distance)
    
    print(f"Initial best distance: {result.best_distance}")
    
    # Evolution loop
    start_time = time.time()
    
    for generation in range(config.maxIter):
        # Calculate population statistics
        best_individual = population.get_best()
        avg_distance = population.get_average_distance()
        diversity_metrics = diversity_manager.calculate_diversity_metrics(population)
        
        # Update best solution if improved
        result.update_best(best_individual.tour, best_individual.distance)
        
        # Update stagnation counter
        diversity_manager.update_stagnation_counter(best_individual.distance)
        
        # Add iteration to history
        result.add_iteration(
            generation,
            best_individual.distance,
            avg_distance,
            diversity_metrics['unique_ratio']
        )
        
        # Call callbacks if provided
        if callbacks:
            callback_state = {
                'iter': generation,
                'best_route': best_individual.tour.copy(),
                'best_distance': best_individual.distance,
                'avg_distance': avg_distance,
                'population': population.individuals.copy(),
                'diversity': diversity_metrics,
                'stagnation': diversity_manager.get_stagnation_level()
            }
            
            for callback in callbacks:
                try:
                    callback(callback_state)
                except Exception as e:
                    print(f"Warning: Callback error in generation {generation}: {e}")
        
        # Print progress periodically
        if generation % 100 == 0 or generation < 10:
            elapsed = time.time() - start_time
            print(f"Gen {generation:4d}: Best={best_individual.distance:8.1f}, "
                  f"Avg={avg_distance:8.1f}, Diversity={diversity_metrics['unique_ratio']:.3f}, "
                  f"Stagnation={diversity_manager.get_stagnation_level()}, "
                  f"Time={elapsed:.1f}s")
        
        # Early stopping if we haven't improved for a very long time
        if diversity_manager.get_stagnation_level() > config.maxIter // 4:
            print(f"Early stopping at generation {generation} due to stagnation")
            break
        
        # Maintain diversity
        population = diversity_manager.maintain_diversity(population, problem)
        
        # Apply adaptive mechanisms if stagnated
        population = diversity_manager.apply_adaptive_mechanisms(population, problem)
        
        # Create new generation
        population = create_new_generation(population, problem, config)
        
        # Remove duplicates periodically
        if generation % 50 == 0:
            population.remove_duplicates(problem)
    
    # Final statistics
    final_best = population.get_best()
    result.update_best(final_best.tour, final_best.distance)
    
    elapsed_time = time.time() - start_time
    print(f"\nGA completed in {elapsed_time:.2f} seconds")
    print(f"Final best distance: {result.best_distance}")
    print(f"Total generations: {len(result.history)}")
    
    return result


# Export main functions as specified in the requirements
def select_tournament(population, k=3):
    """Tournament selection (wrapper for compatibility)."""
    from .selection import select_tournament as _select_tournament
    return _select_tournament(population, k)


def crossover_ox(parent1, parent2):
    """Order crossover (wrapper for compatibility)."""
    from .crossover import crossover_ox as _crossover_ox
    return _crossover_ox(parent1, parent2)


def crossover_pmx(parent1, parent2):
    """Partially mapped crossover (wrapper for compatibility)."""
    from .crossover import crossover_pmx as _crossover_pmx
    return _crossover_pmx(parent1, parent2)


def mutate_swap(individual):
    """Swap mutation (wrapper for compatibility)."""
    from .mutation import mutate_swap as _mutate_swap
    return _mutate_swap(individual)


def mutate_inversion(individual):
    """Inversion mutation (wrapper for compatibility)."""
    from .mutation import mutate_inversion as _mutate_inversion
    return _mutate_inversion(individual)