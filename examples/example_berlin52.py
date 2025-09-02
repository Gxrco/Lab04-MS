# examples/run_berlin52.py
"""
Runner para berlin52 usando tu core GA.
Coloca berlin52.tsp en data/berlin52.tsp (descargar desde TSPLIB).
"""
import os
import sys

# Asegurar que el repo raíz esté en el path al ejecutar directamente este script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import GAConfig, run_ga
from iotsp import load_tsplib_problem, known_optimum


def berlin52_config() -> GAConfig:
    return GAConfig(
        N=600,
        maxIter=3000,
        pct_survivors=0.15,
        pct_crossover=0.55,
        pct_mutation=0.30,
        selection="tournament",
        crossover="OX",
        mutation="inversion",
        elitism=8,
        seed=12345,
    )


def main():
    tsp_path = os.path.join(ROOT, "data", "berlin52.tsp")
    if not os.path.exists(tsp_path):
        print("\n[!] No encontré data/berlin52.tsp\n")
        print("Descárgalo desde TSPLIB y guárdalo como data/berlin52.tsp")
        print("URL: http(s)://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/\n")
        return

    problem = load_tsplib_problem(tsp_path)
    cfg = berlin52_config()

    print(f"Instancia: berlin52 (n={problem.n})")
    print(f"Config: N={cfg.N}, iters={cfg.maxIter}, surv={cfg.pct_survivors}, X={cfg.pct_crossover}, M={cfg.pct_mutation}, elitism={cfg.elitism}")

    result = run_ga(problem, cfg)

    print("\n=== RESULTADOS BERLIN52 ===")
    print(f"Mejor distancia encontrada: {result.best_distance}")

    opt = known_optimum("berlin52")
    if opt is not None:
        gap = result.best_distance - opt
        sign = "+" if gap >= 0 else ""
        print(f"Óptimo conocido: {opt}  |  GAP: {sign}{gap}")

    # Verificación directa
    calc = problem.calculate_tour_distance(result.best_route)
    print(f"Verificación distancia de la ruta: {calc}  |  OK: {abs(calc - result.best_distance) < 1e-6}")


if __name__ == "__main__":
    main()
