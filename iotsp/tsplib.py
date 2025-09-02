from __future__ import annotations
from typing import List, Tuple, Optional
import os

try:
    from ..core.problem import TSPProblem
except Exception:
    from core.problem import TSPProblem


SUPPORTED_TYPES = {"TSP"}
SUPPORTED_WEIGHT_TYPES = {"EUC_2D"}


def parse_tsplib(filepath: str) -> dict:
    """Parsea un archivo .tsp (TSPLIB) básico con NODE_COORD_SECTION.

    Soporta:
      - TYPE: TSP
      - EDGE_WEIGHT_TYPE: EUC_2D
      - NODE_COORD_SECTION

    Retorna un dict con: name, type, dimension, edge_weight_type, coords (List[Tuple[float,float]]).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    name: Optional[str] = None
    tp: Optional[str] = None
    dim: Optional[int] = None
    ew_type: Optional[str] = None
    coords: List[Tuple[float, float]] = []

    in_coords = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            up = line.upper()

            if up.startswith("NAME"):
                # NAME: berlin52
                parts = line.split(":", 1)
                if len(parts) == 2:
                    name = parts[1].strip()
                else:
                    name = line.split()[1]
            elif up.startswith("TYPE"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    tp = parts[1].strip().upper()
            elif up.startswith("DIMENSION"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    dim = int(parts[1].strip())
            elif up.startswith("EDGE_WEIGHT_TYPE"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    ew_type = parts[1].strip().upper()
            elif up.startswith("NODE_COORD_SECTION"):
                in_coords = True
            elif up.startswith("EOF"):
                break
            elif in_coords:
                # Formato típico:  idx  x  y
                parts = line.split()
                if len(parts) >= 3:
                    # ignoramos el índice TSPLIB (1..n) y guardamos (x,y)
                    try:
                        # algunas instancias usan enteros, otras float
                        # parts[0] = id, parts[1] = x, parts[2] = y
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append((x, y))
                    except ValueError:
                        # línea no válida, la saltamos
                        pass
                # si ya reunimos 'dim' coords, podemos terminar
                if dim is not None and len(coords) >= dim:
                    in_coords = False

    # Validaciones básicas
    if tp is None or tp not in SUPPORTED_TYPES:
        raise ValueError(f"TYPE no soportado o ausente: {tp}")
    if ew_type is None or ew_type not in SUPPORTED_WEIGHT_TYPES:
        raise ValueError(f"EDGE_WEIGHT_TYPE no soportado o ausente: {ew_type}")
    if dim is None:
        raise ValueError("DIMENSION ausente en el .tsp")
    if len(coords) != dim:
        raise ValueError(f"Se esperaban {dim} coords, se leyeron {len(coords)}")

    return {
        "name": name or "Unknown",
        "type": tp,
        "dimension": dim,
        "edge_weight_type": ew_type,
        "coords": coords,
    }


def load_tsplib_problem(filepath: str) -> TSPProblem:
    """Carga un .tsp EUC_2D y devuelve un TSPProblem (usa coords)."""
    meta = parse_tsplib(filepath)
    return TSPProblem(coords=meta["coords"])


def known_optimum(name: str) -> Optional[int]:
    """Devuelve el óptimo conocido (si lo tenemos hardcodeado)."""
    key = (name or "").strip().lower()
    if key in {"berlin52", "berlin_52", "berlin-52"}:
        return 7542
    return None
