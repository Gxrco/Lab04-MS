"""
Result classes for genetic algorithm.
"""
from typing import Any


class GAResult:
    """Results from genetic algorithm execution."""
    
    def __init__(self):
        """Initialize empty result."""
        self.best_route: list[int] = []
        self.best_distance: float | int = float('inf')
        self.history: list[dict[str, Any]] = []
    
    def update_best(self, route: list[int], distance: float | int):
        """Update best solution found."""
        if distance < self.best_distance:
            self.best_route = route.copy()
            self.best_distance = distance
    
    def add_iteration(self, iter_num: int, best_distance: float | int, 
                     avg_distance: float | int, diversity: float):
        """Add iteration statistics to history."""
        self.history.append({
            'iter': iter_num,
            'best': best_distance,
            'avg': avg_distance,
            'diversity': diversity
        })