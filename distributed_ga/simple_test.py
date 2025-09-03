#!/usr/bin/env python3
"""
Simple test experiment for distributed GA.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from genetic_algorithm import GeneticAlgorithm
from test_functions import TestFunctions
import numpy as np

def main():
    print("Running simple GA test...")
    
    # Create GA
    ga = GeneticAlgorithm(
        fitness_function=TestFunctions.sphere,
        dimensions=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        population_size=20
    )
    
    print(f"Initial population shape: {ga.population.shape}")
    print(f"Initial bounds: {ga.bounds}")
    
    # Run optimization
    result = ga.run(max_generations=10)
    
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Best solution: {result['best_individual']}")
    print(f"Generations: {result['generations_run']}")
    
    print("Test completed successfully!")

if __name__ == '__main__':
    main()