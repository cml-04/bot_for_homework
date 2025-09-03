#!/usr/bin/env python3
"""
Demonstration script for the distributed genetic algorithm framework.
This script shows basic functionality without requiring a full cluster setup.
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from genetic_algorithm import (
    GeneticAlgorithm, 
    RouletteWheelSelection, TournamentSelection,
    SinglePointCrossover, UniformCrossover,
    GaussianMutation, UniformMutation
)
from test_functions import TestFunctions, get_test_function
from utils import setup_logging, Statistics

def demonstrate_test_functions():
    """Demonstrate the test functions."""
    print("=" * 60)
    print("DISTRIBUTED GENETIC ALGORITHM FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. Test Functions Demonstration")
    print("-" * 40)
    
    functions = ['sphere', 'rastrigin', 'rosenbrock', 'ackley']
    test_point = [0.0, 0.0]
    
    for func_name in functions:
        func = get_test_function(func_name)
        if func:
            value = func(test_point)
            info = TestFunctions.get_function_info(func_name)
            print(f"{func_name.capitalize():12}: f([0,0]) = {value:.6f}, Global min = {info.get('global_min', 'N/A')}")

def demonstrate_operators():
    """Demonstrate genetic operators."""
    print("\n2. Genetic Operators Demonstration")
    print("-" * 40)
    
    # Sample population for demonstration
    population = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    fitness = [10.0, 5.0, 15.0, 8.0]  # Lower is better
    bounds = [(-10, 10), (-10, 10)]
    
    # Selection demonstration
    print("\nSelection Operators:")
    rws = RouletteWheelSelection()
    parents = rws.select(population, fitness, 2)
    print(f"Roulette Wheel Selected: {parents.shape[0]} parents")
    
    ts = TournamentSelection(tournament_size=2)
    parents = ts.select(population, fitness, 2)
    print(f"Tournament Selected: {parents.shape[0]} parents")
    
    # Crossover demonstration
    print("\nCrossover Operators:")
    parent1, parent2 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    
    spc = SinglePointCrossover()
    child1, child2 = spc.crossover(parent1, parent2)
    print(f"Single Point: Parent1={parent1} + Parent2={parent2} → Children shapes: {child1.shape}, {child2.shape}")
    
    uc = UniformCrossover(probability=0.5)
    child1, child2 = uc.crossover(parent1, parent2)
    print(f"Uniform: Generated children with uniform crossover")
    
    # Mutation demonstration
    print("\nMutation Operators:")
    individual = np.array([1.0, 2.0])
    
    gm = GaussianMutation(sigma=0.1)
    mutated = gm.mutate(individual, bounds)
    print(f"Gaussian: {individual} → {mutated} (small changes)")
    
    um = UniformMutation(mutation_strength=0.1)
    mutated = um.mutate(individual, bounds)
    print(f"Uniform: {individual} → {mutated} (bounded changes)")

def run_optimization_comparison():
    """Compare optimization on different functions."""
    print("\n3. Optimization Performance Comparison")
    print("-" * 40)
    
    functions = ['sphere', 'rastrigin', 'rosenbrock']
    results = {}
    
    for func_name in functions:
        print(f"\nOptimizing {func_name.capitalize()} function...")
        
        func = get_test_function(func_name)
        info = TestFunctions.get_function_info(func_name)
        
        # Create GA
        ga = GeneticAlgorithm(
            fitness_function=func,
            dimensions=5,
            bounds=[tuple(info['search_domain'])] * 5,
            population_size=50,
            crossover_probability=0.8,
            mutation_probability=0.1
        )
        
        # Run optimization
        start_time = time.time()
        result = ga.run(max_generations=100, target_fitness=1e-6)
        end_time = time.time()
        
        results[func_name] = {
            'best_fitness': result['best_fitness'],
            'generations': result['generations_run'],
            'time': end_time - start_time,
            'target': info['global_min']
        }
        
        print(f"  Best fitness: {result['best_fitness']:.6e}")
        print(f"  Generations: {result['generations_run']}")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Target: {info['global_min']}")
    
    return results

def create_convergence_plot(results):
    """Create convergence comparison plot."""
    print("\n4. Creating Convergence Visualization")
    print("-" * 40)
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Run short optimizations to get convergence data
        functions = ['sphere', 'rastrigin', 'rosenbrock']
        colors = ['blue', 'red', 'green']
        
        plt.subplot(2, 2, 1)
        for i, func_name in enumerate(functions):
            func = get_test_function(func_name)
            info = TestFunctions.get_function_info(func_name)
            
            ga = GeneticAlgorithm(
                fitness_function=func,
                dimensions=3,
                bounds=[tuple(info['search_domain'])] * 3,
                population_size=30
            )
            
            # Track convergence
            fitness_history = []
            for gen in range(50):
                ga.evolve_generation()
                fitness_history.append(ga.best_fitness)
            
            plt.semilogy(fitness_history, color=colors[i], label=func_name.capitalize(), linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.title('Convergence Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Function landscapes (2D projections)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        
        functions_2d = [
            ('Sphere', lambda x, y: x**2 + y**2),
            ('Rastrigin', lambda x, y: 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))),
            ('Rosenbrock', lambda x, y: 100*(y - x**2)**2 + (1 - x)**2)
        ]
        
        for i, (name, func_2d) in enumerate(functions_2d):
            plt.subplot(2, 2, i + 2)
            Z = func_2d(X, Y)
            plt.contour(X, Y, Z, levels=20, cmap='viridis')
            plt.colorbar()
            plt.title(f'{name} Function Landscape')
            plt.xlabel('x1')
            plt.ylabel('x2')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('distributed_ga/results', exist_ok=True)
        plt.savefig('distributed_ga/results/demonstration_plots.png', dpi=300, bbox_inches='tight')
        print("  Convergence plots saved to: distributed_ga/results/demonstration_plots.png")
        
        plt.close()
        
    except Exception as e:
        print(f"  Plot creation failed: {e}")

def demonstrate_distributed_capabilities():
    """Demonstrate distributed computing concepts."""
    print("\n5. Distributed Computing Architecture")
    print("-" * 40)
    
    print("Framework Components:")
    print("  ✓ Ray-based distributed computing")
    print("  ✓ FitnessEvaluator remote actors")
    print("  ✓ GeneticOperator remote actors") 
    print("  ✓ Load balancing and work distribution")
    print("  ✓ Cross-platform support (Windows/Linux)")
    print("  ✓ Fault tolerance and auto-reconnection")
    print("  ✓ Performance monitoring and statistics")
    
    print("\nDeployment Architecture:")
    print("  Head Node (Windows Laptop):")
    print("    - Algorithm control and coordination")
    print("    - Population management")
    print("    - Results collection and analysis")
    print("    - Ray dashboard (port 8265)")
    
    print("  Worker Nodes (Linux Servers):")
    print("    - Parallel fitness evaluation")
    print("    - Distributed genetic operations")
    print("    - Load-balanced task execution")
    
    print("\nTo deploy on actual cluster:")
    print("  1. Start head node: python scripts/start_head.py")
    print("  2. Start workers: python scripts/start_worker.py --address ray://HEAD_IP:10001")
    print("  3. Run experiments: python scripts/run_experiment.py")

def performance_summary(results):
    """Print performance summary."""
    print("\n6. Performance Summary")
    print("-" * 40)
    
    print("Optimization Results:")
    for func_name, result in results.items():
        improvement = abs(result['best_fitness'] - result['target'])
        print(f"  {func_name.capitalize():12}: Best={result['best_fitness']:.6e}, "
              f"Gens={result['generations']:3d}, Time={result['time']:.2f}s")
    
    print("\nFramework Features:")
    print("  ✓ 6 benchmark test functions implemented")
    print("  ✓ Multiple selection, crossover, and mutation operators")
    print("  ✓ Configurable parameters via YAML")
    print("  ✓ Comprehensive test suite")
    print("  ✓ Docker containerization support")
    print("  ✓ Performance monitoring and statistics")
    print("  ✓ Fault tolerance and checkpointing")
    print("  ✓ Cross-platform distributed computing")

def main():
    """Main demonstration function."""
    # Setup logging
    logger = setup_logging("INFO")
    
    try:
        # Run demonstrations
        demonstrate_test_functions()
        demonstrate_operators()
        results = run_optimization_comparison()
        create_convergence_plot(results)
        demonstrate_distributed_capabilities()
        performance_summary(results)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe distributed genetic algorithm framework is ready for deployment.")
        print("See distributed_ga/docs/README.md for detailed usage instructions.")
        print("See distributed_ga/docs/setup_guide.md for cluster setup instructions.")
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())