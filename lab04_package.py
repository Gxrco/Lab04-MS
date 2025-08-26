"""
MS-Lab04: Genetic Algorithm for TSP
Modelación y Simulación 2025 - Lab 04

A modular implementation of genetic algorithms for solving the 
Traveling Salesman Problem (TSP).

Modules:
- core: Core GA implementation
- io: I/O and data management
- viz: Visualization and animations
"""

from . import core
from . import io
from . import viz
from . import config
from . import data

# Re-export main classes for convenience
from .core import TSPProblem, GAConfig, GAResult, run_ga

__version__ = "1.0.0"

__all__ = [
    'core',
    'io', 
    'viz',
    'config',
    'data',
    # Main classes
    'TSPProblem',
    'GAConfig',
    'GAResult',
    'run_ga',
]