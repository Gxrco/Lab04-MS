"""
Configuration classes for genetic algorithm.
"""
from typing import Optional


class GAConfig:
    """Configuration parameters for genetic algorithm."""
    
    def __init__(self, N: int, maxIter: int, pct_survivors: float, 
                 pct_crossover: float, pct_mutation: float,
                 selection: str = "tournament", crossover: str = "OX",
                 mutation: str = "inversion", elitism: int = 1,
                 seed: Optional[int] = None):
        """
        Initialize GA configuration.
        
        Args:
            N: Population size
            maxIter: Maximum number of iterations
            pct_survivors: Percentage of population selected as survivors
            pct_crossover: Percentage of population created by crossover
            pct_mutation: Percentage of population created by mutation
            selection: Selection method ("tournament" or "rank")
            crossover: Crossover method ("OX" or "PMX")
            mutation: Mutation method ("swap" or "inversion")
            elitism: Number of elite individuals to preserve
            seed: Random seed for reproducibility
        """
        self.N = N
        self.maxIter = maxIter
        self.pct_survivors = pct_survivors
        self.pct_crossover = pct_crossover
        self.pct_mutation = pct_mutation
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.elitism = elitism
        self.seed = seed
        
        # Validate percentages sum approximately to 1.0
        total_pct = pct_survivors + pct_crossover + pct_mutation
        if abs(total_pct - 1.0) > 0.01:
            raise ValueError(f"Percentages must sum to ~1.0, got {total_pct}")
        
        # Validate selection method
        if selection not in ["tournament", "rank"]:
            raise ValueError(f"Invalid selection method: {selection}")
        
        # Validate crossover method
        if crossover not in ["OX", "PMX"]:
            raise ValueError(f"Invalid crossover method: {crossover}")
        
        # Validate mutation method
        if mutation not in ["swap", "inversion"]:
            raise ValueError(f"Invalid mutation method: {mutation}")
    
    def get_num_survivors(self) -> int:
        """Get number of survivors."""
        return int(self.N * self.pct_survivors)
    
    def get_num_crossover(self) -> int:
        """Get number of offspring from crossover."""
        return int(self.N * self.pct_crossover)
    
    def get_num_mutation(self) -> int:
        """Get number of offspring from mutation."""
        return int(self.N * self.pct_mutation)