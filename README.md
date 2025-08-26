# MS-Lab04: Genetic Algorithm for TSP

**Modelación y Simulación 2025 - Lab 04**  
*26.agosto.2025*

A modular implementation of genetic algorithms for solving the Traveling Salesman Problem (TSP).

## Project Structure

```
MS-Lab04/
├── core/           # 🧬 Core GA Implementation (Person A)
│   ├── problem.py      # TSP problem representation
│   ├── config.py       # GA configuration
│   ├── result.py       # GA results
│   ├── individual.py   # Individual representation
│   ├── population.py   # Population management
│   ├── selection.py    # Selection operators
│   ├── crossover.py    # Crossover operators (OX, PMX)
│   ├── mutation.py     # Mutation operators (swap, inversion)
│   ├── replacement.py  # Replacement strategies
│   ├── diversity.py    # Diversity control & adaptive mechanisms
│   └── ga_core.py      # Main GA algorithm
│
├── io/             # 📁 I/O & Data Management (Person B)
│   └── (To be implemented)
│
├── viz/            # 📊 Visualization (Person C)
│   └── (To be implemented)
│
├── config/         # ⚙️  Shared configurations
├── data/           # 📊 TSP datasets (Berlin52, etc.)
├── tests/          # 🧪 Unit tests
├── examples/       # 💡 Usage examples
└── docs/           # 📚 Documentation
```

## Quick Start

### Running the Example

```bash
python examples/example_usage.py
```

### Running Tests

```bash
python -m pytest tests/test_operators.py -v
```

## Core Module (Person A) - ✅ COMPLETED

### Main Components

- **TSPProblem**: Handles coordinates or distance matrices
- **GAConfig**: Configuration with all GA parameters
- **Individual**: Tour representation with fitness evaluation  
- **Population**: Population management with diversity control
- **Operators**: Selection, crossover (OX, PMX), mutation (swap, inversion)
- **GA Core**: Main algorithm with adaptive mechanisms

### Key Features

✅ **Correct Operators**: All operators produce valid permutations  
✅ **Multiple Crossovers**: OX (Order) and PMX (Partially Mapped)  
✅ **Adaptive Mechanisms**: Stagnation detection, diversity control  
✅ **Elitism**: Preserves best individuals  
✅ **Comprehensive Tests**: 15/15 unit tests pass  

### Usage

```python
from core import TSPProblem, GAConfig, run_ga

# Create problem
coords = [(0,0), (1,0), (1,1), (0,1)]
problem = TSPProblem(coords=coords)

# Configure GA
config = GAConfig(
    N=100,                   # Population size
    maxIter=1000,           # Max generations
    pct_survivors=0.3,      # 30% survivors
    pct_crossover=0.5,      # 50% from crossover
    pct_mutation=0.2,       # 20% from mutation
    selection="tournament",
    crossover="OX", 
    mutation="inversion",
    elitism=2,
    seed=42
)

# Run algorithm
result = run_ga(problem, config)
print(f"Best distance: {result.best_distance}")
print(f"Best tour: {result.best_route}")
```

## Integration Points

### For Person B (I/O & Experiments)

```python
from core import TSPProblem, GAConfig, run_ga

# Your parsers should create TSPProblem instances
# Your experiment runners should use run_ga()
```

### For Person C (Visualization)

```python
from core import run_ga

# Use callbacks parameter for real-time visualization
def viz_callback(state):
    # state contains: iter, best_route, best_distance, population, etc.
    pass

result = run_ga(problem, config, callbacks=[viz_callback])
```

## Berlin52 Target

- **Optimal Distance**: ≈ 7542
- **Target**: ≤ 7800 consistently
- **Current Status**: Ready for integration

---

## Development Guidelines

1. **Core Module**: ✅ Complete and tested
2. **I/O Module**: Use `core.TSPProblem` for data loading
3. **Viz Module**: Use callbacks for real-time updates  
4. **Testing**: Run `pytest` before commits
5. **Imports**: Use direct imports (`from core import ...`)

## Files Overview

| Directory | Status | Responsibility |
|-----------|--------|---------------|
| `core/`   | ✅ Complete | Person A - GA algorithms |
| `io/`     | 📝 Pending  | Person B - Data & experiments |
| `viz/`    | 📝 Pending  | Person C - Visualization |
| `tests/`  | ✅ Working  | Unit tests (15 passing) |
| `examples/` | ✅ Working | Usage examples |