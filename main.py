"""
Main file to run the TSP Genetic Algorithm with visualization.
This file integrates all GA components and visualization.
"""

import sys
import os
import numpy as np
from typing import List, Tuple

from core.problem import TSPProblem
from core.config import GAConfig
from core.ga_core import run_ga

from viz.tsp_visualizer import TSPVisualizer

def get_ga_config(instance_size: int) -> GAConfig:
    """
    Gets appropriate configuration based on problem size.
    
    Args:
        instance_size: Number of cities
        
    Returns:
        GA Configuration
    """
    if instance_size <= 10:
        return GAConfig(
            N=50,                    
            maxIter=200,            
            pct_survivors=0.4,      
            pct_crossover=0.4,      
            pct_mutation=0.2,       
            selection="tournament",
            crossover="OX",
            mutation="inversion",
            elitism=2,
            seed=42
        )
    elif instance_size <= 16:
        return GAConfig(
            N=100,                  
            maxIter=400,          
            pct_survivors=0.3,   
            pct_crossover=0.5,     
            pct_mutation=0.2,      
            selection="tournament",
            crossover="OX",
            mutation="inversion",
            elitism=3,
            seed=42
        )
    else:
        return GAConfig(
            N=150,                 
            maxIter=600,          
            pct_survivors=0.25,   
            pct_crossover=0.55,  
            pct_mutation=0.2,     
            selection="tournament",
            crossover="OX",
            mutation="inversion",
            elitism=5,
            seed=42
        )

def input_coordinates_manually() -> List[Tuple[float, float]]:
    """
    Allows user to input coordinates manually.
    
    Returns:
        List of coordinates input by the user
    """
    coords = []
    print("\nIngreso manual de coordenadas")
    print("Formato: x,y (ejemplo: 10.5,20.3)")
    print("Ingresa 'fin' cuando termines de agregar ciudades")
    
    city_num = 1
    while True:
        try:
            coord_input = input(f"Ciudad {city_num} (x,y): ").strip()
            
            if coord_input.lower() in ['fin', 'end', 'terminar', '']:
                if len(coords) < 3:
                    print("Necesitas al menos 3 ciudades para el TSP.")
                    continue
                break
            
            if ',' in coord_input:
                x_str, y_str = coord_input.split(',', 1)
                x = float(x_str.strip())
                y = float(y_str.strip())
                coords.append((x, y))
                city_num += 1
            else:
                print("Formato incorrecto. Usa: x,y")
                
        except ValueError:
            print("Coordenadas inválidas. Usa números separados por coma.")
        except KeyboardInterrupt:
            print("\nOperación cancelada.")
            return []
    
    print(f"Se ingresaron {len(coords)} ciudades.")
    return coords

def generate_random_coordinates() -> List[Tuple[float, float]]:
    """
    Generates random coordinates with user parameters.
    
    Returns:
        List of generated coordinates
    """
    print("\nGeneración de coordenadas aleatorias")
    
    try:
        num_cities = int(input("Número de ciudades (3-60): "))
        if num_cities < 3 or num_cities > 60:
            print("Número fuera de rango. Usando 15 ciudades.")
            num_cities = 15
            
        min_coord = float(input("Coordenada mínima (default 0): ") or "0")
        max_coord = float(input("Coordenada máxima (default 200): ") or "200")
        
        if min_coord >= max_coord:
            print("Rango inválido. Usando 0-200.")
            min_coord, max_coord = 0, 200
            
        seed = input("Semilla aleatoria (Enter para aleatoria): ").strip()
        if seed:
            np.random.seed(int(seed))
        
        coords = []
        for i in range(num_cities):
            x = np.random.uniform(min_coord, max_coord)
            y = np.random.uniform(min_coord, max_coord)
            coords.append((round(x, 2), round(y, 2)))
        
        print(f"Generadas {num_cities} ciudades en rango [{min_coord}, {max_coord}]")
        return coords
        
    except ValueError:
        print("Valores inválidos. Generando 15 ciudades por defecto.")
        coords = []
        for _ in range(15):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            coords.append((round(x, 2), round(y, 2)))
        return coords


def configure_ga_parameters(num_cities: int) -> GAConfig:
    """
    Allows user to configure genetic algorithm parameters.
    
    Args:
        num_cities: Number of cities to suggest default values
        
    Returns:
        Custom GA configuration
    """
    print(f"\nConfiguración del Algoritmo Genético ({num_cities} ciudades)")
    print("Presiona Enter para usar valores sugeridos")
    
    suggested_pop = max(50, min(200, num_cities * 8))
    suggested_iter = max(200, min(800, num_cities * 30))
    
    try:
        pop_size = input(f"Tamaño de población (sugerido {suggested_pop}): ").strip()
        pop_size = int(pop_size) if pop_size else suggested_pop
        
        max_iter = input(f"Iteraciones máximas (sugerido {suggested_iter}): ").strip()
        max_iter = int(max_iter) if max_iter else suggested_iter
        
        survivors = input("% Supervivientes (sugerido 30): ").strip()
        survivors = float(survivors)/100 if survivors else 0.3
        
        crossover = input("% Crossover (sugerido 50): ").strip()
        crossover = float(crossover)/100 if crossover else 0.5
        
        mutation = input("% Mutación (sugerido 20): ").strip()
        mutation = float(mutation)/100 if mutation else 0.2
        
        total = survivors + crossover + mutation
        if abs(total - 1.0) > 0.01:
            print(f"Advertencia: Los porcentajes suman {total*100:.1f}%, se normalizarán automáticamente")
        
        elitism = input("Individuos élite (sugerido 3): ").strip()
        elitism = int(elitism) if elitism else 3
        
        seed = input("Semilla aleatoria (Enter para 42): ").strip()
        seed = int(seed) if seed else 42
        
        return GAConfig(
            N=pop_size,
            maxIter=max_iter,
            pct_survivors=survivors,
            pct_crossover=crossover,
            pct_mutation=mutation,
            selection="tournament",
            crossover="OX",
            mutation="inversion",
            elitism=elitism,
            seed=seed
        )
        
    except ValueError as e:
        print(f"Error en configuración: {e}")
        print("Usando configuración por defecto...")
        return get_ga_config(num_cities)

def main():    
    print("=" * 60)
    print("ALGORITMO GENÉTICO PARA TSP - CON VISUALIZACIÓN")
    print("=" * 60)
    
    coords = get_user_coordinates()
    if not coords:
        print("No se obtuvieron coordenadas válidas. Terminando...")
        return
    
    print(f"\nProblema TSP creado con {len(coords)} ciudades")
    print("Coordenadas:")
    for i, (x, y) in enumerate(coords):
        print(f"  Ciudad {i}: ({x}, {y})")
    
    config = get_ga_configuration(len(coords))
    
    print(f"\nConfiguración del GA:")
    print(f"  Población: {config.N}")
    print(f"  Iteraciones máximas: {config.maxIter}")
    print(f"  Supervivientes: {config.pct_survivors*100:.0f}%")
    print(f"  Crossover: {config.pct_crossover*100:.0f}%")
    print(f"  Mutación: {config.pct_mutation*100:.0f}%")
    print(f"  Elitismo: {config.elitism}")
    print(f"  Selección: {config.selection}")
    print(f"  Método crossover: {config.crossover}")
    print(f"  Método mutación: {config.mutation}")
    print(f"  Semilla: {config.seed}")
    
    problem = TSPProblem(coords=coords)
    
    show_viz = input("\n¿Mostrar visualización en tiempo real? (s/n, default=s): ").strip().lower()
    show_viz = show_viz in ['s', 'si', 'yes', ''] or show_viz == 'y'
    
    visualizer = None
    callbacks = []
    
    if show_viz:
        print("Preparando visualización...")
        visualizer = TSPVisualizer(coords, figsize=(15, 8))
        callbacks = [visualizer.callback_function]
    
    print("\nEjecutando algoritmo genético...")
    print("=" * 40)
    
    try:
        result = run_ga(
            problem=problem,
            config=config,
            callbacks=callbacks
        )
        
        print("\n" + "=" * 60)
        print("RESULTADOS FINALES")
        print("=" * 60)
        print(f"Mejor ruta encontrada: {result.best_route}")
        print(f"Mejor distancia: {result.best_distance}")
        print(f"Generaciones ejecutadas: {len(result.history)}")
        
        if result.history:
            print("\nEvolución de la convergencia:")
            indices_to_show = [0]
            if len(result.history) > 10:
                indices_to_show.extend([10, 25, 50, 75, 100])
            indices_to_show.append(len(result.history) - 1)
            
            for i in indices_to_show:
                if i < len(result.history):
                    hist = result.history[i]
                    print(f"Gen {hist['iter']:3d}: Mejor={hist['best']:6.1f}, "
                          f"Promedio={hist['avg']:6.1f}, Diversidad={hist['diversity']:.3f}")
        
        calculated_distance = problem.calculate_tour_distance(result.best_route)
        print(f"\nVerificación:")
        print(f"  Distancia calculada: {calculated_distance}")
        print(f"  Coincide con resultado: {abs(calculated_distance - result.best_distance) < 0.001}")
        
        if visualizer:
            print(f"\nMostrando visualización final...")
            print("Cierra la ventana para terminar el programa.")
            
            save_anim = input("¿Guardar animación como GIF? (s/n): ").strip().lower()
            if save_anim in ['s', 'si', 'yes', 'y']:
                filename = input("Nombre del archivo (sin .gif): ").strip() or "tsp_resultado"
                print(f"Guardando animación como {filename}.gif...")
                visualizer.save_animation(f"{filename}.gif", fps=8)
                print("Animación guardada.")
            
            visualizer.show_final_result()
        
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario.")
        if visualizer and len(visualizer.route_history) > 0:
            print("Mostrando resultados parciales...")
            visualizer.show_final_result()
    
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


def get_user_coordinates() -> List[Tuple[float, float]]:
    """Gets coordinates from user through different methods."""
    
    print("\nSelecciona cómo ingresar las coordenadas:")
    print("1. Escribir coordenadas manualmente")
    print("2. Generar aleatoriamente")
    print("3. Usar ejemplo de prueba")
    
    while True:
        choice = input("Opción (1-4): ").strip()
        
        if choice == '1':
            return input_coordinates_manually()
        elif choice == '2':
            return generate_random_coordinates()
        elif choice == '3':
            return get_test_example()
        else:
            print("Opción inválida. Selecciona 1, 2, 3 o 4.")

def get_test_example() -> List[Tuple[float, float]]:

    examples = {
        '1': {
            'name': 'Cuadrado simple (8 ciudades)',
            'coords': [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        },
        '2': {
            'name': 'Círculo (10 ciudades)',
            'coords': [(50 + 25*np.cos(2*np.pi*i/10), 50 + 25*np.sin(2*np.pi*i/10)) for i in range(10)]
        },
        '3': {
            'name': 'Aleatorio (12 ciudades)',
            'coords': [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(12)]
        }
    }
    
    print("\nEjemplos disponibles:")
    for key, example in examples.items():
        print(f"{key}. {example['name']}")
    
    while True:
        choice = input("Selecciona ejemplo (1-3): ").strip()
        if choice in examples:
            coords = examples[choice]['coords']
            coords = [(round(x, 2), round(y, 2)) for x, y in coords]
            print(f"Seleccionado: {examples[choice]['name']}")
            return coords
        print("Selección inválida.")

def get_ga_configuration(num_cities: int) -> GAConfig:

    print(f"\nConfiguración del Algoritmo Genético:")
    print("Presiona Enter para usar valores por defecto")
    
    default_pop = max(50, min(200, num_cities * 8))
    default_iter = max(100, min(500, num_cities * 20))
    
    try:
        pop_str = input(f"Tamaño de población (default {default_pop}): ").strip()
        pop_size = int(pop_str) if pop_str else default_pop
        
        iter_str = input(f"Iteraciones máximas (default {default_iter}): ").strip()
        max_iter = int(iter_str) if iter_str else default_iter
        
        advanced = input("¿Configurar parámetros avanzados? (s/n): ").strip().lower()
        
        if advanced in ['s', 'si', 'yes', 'y']:
            survivors = float(input("% Supervivientes (default 30): ") or "30") / 100
            crossover = float(input("% Crossover (default 50): ") or "50") / 100
            mutation = float(input("% Mutación (default 20): ") or "20") / 100
            elitism = int(input("Individuos élite (default 3): ") or "3")
            seed = int(input("Semilla aleatoria (default 42): ") or "42")
        else:
            survivors, crossover, mutation = 0.3, 0.5, 0.2
            elitism, seed = 3, 42
        
        return GAConfig(
            N=pop_size,
            maxIter=max_iter,
            pct_survivors=survivors,
            pct_crossover=crossover,
            pct_mutation=mutation,
            selection="tournament",
            crossover="OX",
            mutation="inversion",
            elitism=elitism,
            seed=seed
        )
        
    except ValueError:
        print("Entrada inválida, usando configuración por defecto.")
        return GAConfig(
            N=default_pop,
            maxIter=default_iter,
            pct_survivors=0.3,
            pct_crossover=0.5,
            pct_mutation=0.2,
            selection="tournament",
            crossover="OX",
            mutation="inversion",
            elitism=3,
            seed=42
        )

if __name__ == "__main__":
    main()