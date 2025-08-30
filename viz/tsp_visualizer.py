"""
TSP Genetic Algorithm Visualizer
This module creates a visual simulation that shows how 
the best route found by the genetic algorithm evolves.
"""

import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import matplotlib.pyplot as plt

class TSPVisualizer:
    """TSP Visualizer for a genetic algorithm"""
    
    def __init__(self, coords: List[Tuple[float, float]], figsize: Tuple[int, int] = (12, 8)):
        self.coords = np.array(coords, dtype=float)
        self.n_cities = len(coords)

        self.fig, (self.ax_map, self.ax_stats) = plt.subplots(1, 2, figsize=figsize)
        self.fig.suptitle('Algoritmo Genético para TSP - Evolución en Tiempo Real', fontsize=16)

        dx = np.ptp(self.coords[:, 0]) if self.n_cities > 0 else 0.0
        dy = np.ptp(self.coords[:, 1]) if self.n_cities > 0 else 0.0
        pad = max(dx, dy) * 0.05
        if pad == 0.0:
            pad = 1.0 

        plt.ion()

        self.route_history = []
        self.distance_history = []
        self.avg_distance_history = []
        self.diversity_history = []
        self.generation_history = []

        self.route_line = None
        self.city_points = None
        self.best_distance_line = None
        self.avg_distance_line = None
        self.diversity_line = None

        self._setup_plots()

        self.ax_map.set_xlim(self.coords[:,0].min() - pad, self.coords[:,0].max() + pad)
        self.ax_map.set_ylim(self.coords[:,1].min() - pad, self.coords[:,1].max() + pad)
        
    def _setup_plots(self):
        self.ax_map.set_title('Mejor Ruta Actual')
        self.ax_map.set_xlabel('Coordenada X')
        self.ax_map.set_ylabel('Coordenada Y')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_aspect('equal')
        
        self.city_points = self.ax_map.scatter(
            self.coords[:, 0], self.coords[:, 1], 
            c='red', s=100, zorder=5, alpha=0.8
        )
        
        for i, (x, y) in enumerate(self.coords):
            self.ax_map.annotate(str(i), (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        self.route_line, = self.ax_map.plot([], [], 'b-', linewidth=2, alpha=0.7)
        
        self.ax_stats.set_title('Evolución del Algoritmo')
        self.ax_stats.set_xlabel('Generación')
        self.ax_stats.set_ylabel('Distancia / Diversidad')
        self.ax_stats.grid(True, alpha=0.3)
        
        self.best_distance_line, = self.ax_stats.plot([], [], 'g-', label='Mejor Distancia', linewidth=2)
        self.avg_distance_line, = self.ax_stats.plot([], [], 'b--', label='Distancia Promedio', alpha=0.7)
        self.diversity_line, = self.ax_stats.plot([], [], 'r:', label='Diversidad', alpha=0.8)
        
        self.ax_stats.legend()
        
    def callback_function(self, state: Dict[str, Any]):
        """
        Callback function called at each GA iteration.
        
        Args:
            state: Dictionary with the current algorithm state
        """
        self.route_history.append(state['best_route'].copy())
        self.distance_history.append(state['best_distance'])
        self.avg_distance_history.append(state['avg_distance'])
        self.diversity_history.append(state['diversity']['unique_ratio'])
        self.generation_history.append(state['iter'])
        
        if state['iter'] % 5 == 0 or state['iter'] < 10:
            self._update_visualization()
    
    def _update_visualization(self):
        """Updates visualization with the most recent data."""

        if not self.route_history:
            return
            
        current_route = self.route_history[-1]
        route_coords = self.coords[current_route + [current_route[0]]]  
        
        self.route_line.set_data(route_coords[:, 0], route_coords[:, 1])
        
        current_distance = self.distance_history[-1]
        self.ax_map.set_title(f'Mejor Ruta - Distancia: {current_distance:.2f}')
        
        if len(self.generation_history) > 1:
            self.best_distance_line.set_data(self.generation_history, self.distance_history)
            self.avg_distance_line.set_data(self.generation_history, self.avg_distance_history)
            
            scaled_diversity = np.array(self.diversity_history) * max(self.distance_history)
            self.diversity_line.set_data(self.generation_history, scaled_diversity)
            
            self.ax_stats.relim()
            self.ax_stats.autoscale_view()
        
        plt.pause(0.01)  
        
    def create_animation_from_history(self, interval: int = 200) -> animation.FuncAnimation:
        """
        Creates an animation from the stored history.
        
        Args:
            interval: Interval between frames in milliseconds
            
        Returns:
            FuncAnimation object
        """
        def animate(frame):
            if frame >= len(self.route_history):
                return
                
            route = self.route_history[frame]
            route_coords = self.coords[route + [route[0]]]
            self.route_line.set_data(route_coords[:, 0], route_coords[:, 1])
            
            distance = self.distance_history[frame]
            generation = self.generation_history[frame]
            self.ax_map.set_title(f'Gen: {generation}, Distancia: {distance:.2f}')
            
            if frame > 0:
                self.best_distance_line.set_data(
                    self.generation_history[:frame+1], 
                    self.distance_history[:frame+1]
                )
                self.avg_distance_line.set_data(
                    self.generation_history[:frame+1], 
                    self.avg_distance_history[:frame+1]
                )
                
                scaled_diversity = np.array(self.diversity_history[:frame+1]) * max(self.distance_history[:frame+1])
                self.diversity_line.set_data(
                    self.generation_history[:frame+1], 
                    scaled_diversity
                )
        
        anim = animation.FuncAnimation(
            self.fig, animate, frames=len(self.route_history),
            interval=interval, repeat=True, blit=False
        )
        
        return anim
    
    def save_animation(self, filename: str = 'tsp_evolution.gif', fps: int = 5):
        """
        Save animation as GIF.
        
        Args:
            filename: Name of the file
            fps: Frames per second
        """
        anim = self.create_animation_from_history(interval=1000//fps)
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Animación guardada como {filename}")
    
    def show_final_result(self):
        if self.route_history:
            self._update_visualization()
            
            final_distance = self.distance_history[-1]
            final_generation = self.generation_history[-1]
            
            self.fig.suptitle(
                f'TSP Resuelto - Gen: {final_generation}, Distancia Final: {final_distance:.2f}',
                fontsize=16
            )
            
        plt.tight_layout()
        plt.show()

def create_random_tsp_instance(n_cities: int = 20, seed: int = 42) -> List[Tuple[float, float]]:
    """
    Creates a random TSP instance.
    
    Args:
        n_cities: Number of cities
        seed: Seed for reproducibility
        
    Returns:
        List of city coordinates
    """
    np.random.seed(seed)
    coords = []
    for _ in range(n_cities):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        coords.append((x, y))
    return coords