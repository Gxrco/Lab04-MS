# MS-Lab04: Genetic Algorithm for TSP

**Modelación y Simulación 2025 - Lab 04**  
_26.agosto.2025_

A modular implementation of genetic algorithms for solving the Traveling Salesman Problem (TSP).

## Project Structure

```
MS-Lab04/
├── core/           # 🧬 Core GA Implementation
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
├── iotsp/              # 📁 I/O & Data Management
│   └── tsplib.py       # Extract and parse data from Berlin52.tsp
│
├── viz/            # 📊 Visualization
│   └── tsp_visualizer.py # Render in real time the GA and TSP
│
├── config/         # ⚙️  Shared configurations
├── data/           # 📊 TSP datasets (Berlin52, etc.)
├── tests/          # 🧪 Unit tests
├── examples/       # 💡 Usage examples
└── main.py         # Main program for testing TSP with visualization
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
