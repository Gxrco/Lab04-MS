"""
I/O and Data Management Module (Person B)


Funciones para cargar datasets TSPLIB y helpers de experimentos.
"""
from .tsplib import parse_tsplib, load_tsplib_problem, known_optimum


__all__ = [
"parse_tsplib",
"load_tsplib_problem",
"known_optimum",
]