"""
Example usage of the genetic algorithm core for TSP.

This demonstrates how to use the GA core module (Person A's work)
without requiring the I/O or visualization modules.
"""

import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import TSPProblem, GAConfig, run_ga

def main():
    """Run a simple example."""
    # Set seed for reproducibility
    random.seed(42)
    
    # Create a simple TSP problem with 8 cities in a square pattern
    coords = [
        (0, 0),    # City 0
        (1, 0),    # City 1  
        (2, 0),    # City 2
        (2, 1),    # City 3
        (2, 2),    # City 4
        (1, 2),    # City 5
        (0, 2),    # City 6
        (0, 1),    # City 7
    ]
    
    problem = TSPProblem(coords=coords)
    print(f"Created TSP problem with {problem.n} cities")
    
    # Create GA configuration
    config = GAConfig(
        N=50,                    # Population size
        maxIter=100,            # Maximum iterations
        pct_survivors=0.3,      # 30% survivors
        pct_crossover=0.5,      # 50% from crossover
        pct_mutation=0.2,       # 20% from mutation
        selection="tournament",  # Selection method
        crossover="OX",         # Crossover method
        mutation="inversion",   # Mutation method
        elitism=2,              # Keep 2 best individuals
        seed=42                 # Random seed
    )
    
    print(f"Configuration: Pop={config.N}, Iter={config.maxIter}")
    print(f"Selection: {config.selection}, Crossover: {config.crossover}, Mutation: {config.mutation}")
    
    # Run genetic algorithm
    print("\nRunning genetic algorithm...")
    result = run_ga(problem, config)
    
    # Display results
    print(f"\n=== RESULTS ===")
    print(f"Best tour: {result.best_route}")
    print(f"Best distance: {result.best_distance}")
    print(f"Generations run: {len(result.history)}")
    
    # Show convergence
    print("\n=== CONVERGENCE ===")
    for i in [0, 10, 25, 50, 75, len(result.history)-1]:
        if i < len(result.history):
            hist = result.history[i]
            print(f"Gen {hist['iter']:3d}: Best={hist['best']:6.1f}, Avg={hist['avg']:6.1f}, Diversity={hist['diversity']:.3f}")
    
    # Verify solution
    calculated_distance = problem.calculate_tour_distance(result.best_route)
    print(f"\nVerification: Calculated distance = {calculated_distance}")
    print(f"Matches result: {abs(calculated_distance - result.best_distance) < 0.001}")

if __name__ == "__main__":
    main()