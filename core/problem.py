"""
TSP Problem representation for genetic algorithm.
"""
from typing import Optional


class TSPProblem:
    """Represents a TSP problem instance with coordinates or distance matrix."""
    
    def __init__(self, coords: Optional[list[tuple[float, float]]] = None, 
                 dist: Optional[list[list[int | float]]] = None):
        """
        Initialize TSP problem.
        
        Args:
            coords: List of (x, y) coordinates for each city
            dist: Distance matrix (symmetric NxN)
        """
        self.coords = coords
        self.dist = dist
        
        if coords is not None:
            self.n = len(coords)
        elif dist is not None:
            self.n = len(dist)
        else:
            raise ValueError("Either coords or dist must be provided")
        
        # Build distance matrix from coordinates if needed
        if coords is not None and dist is None:
            self.dist = self._build_distance_matrix(coords)
    
    def _build_distance_matrix(self, coords: list[tuple[float, float]]) -> list[list[float]]:
        """Build distance matrix from coordinates using Euclidean distance."""
        n = len(coords)
        dist = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                # Round to nearest integer (TSPLIB convention for EUC_2D)
                distance = round(distance)
                dist[i][j] = dist[j][i] = distance
        
        return dist
    
    def get_distance(self, city1: int, city2: int) -> float | int:
        """Get distance between two cities."""
        return self.dist[city1][city2]
    
    def calculate_tour_distance(self, tour: list[int]) -> float | int:
        """Calculate total distance of a tour (closed loop)."""
        total_dist = 0
        n = len(tour)
        for i in range(n):
            total_dist += self.get_distance(tour[i], tour[(i + 1) % n])
        return total_dist