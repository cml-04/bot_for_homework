"""
Unit tests for genetic algorithm components.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from genetic_algorithm import (
    RouletteWheelSelection, TournamentSelection,
    SinglePointCrossover, UniformCrossover, ArithmeticCrossover,
    GaussianMutation, UniformMutation,
    GeneticAlgorithm
)
from test_functions import TestFunctions, get_test_function


class TestSelectionOperators:
    """Test selection operators."""
    
    def test_roulette_wheel_selection(self):
        """Test roulette wheel selection."""
        population = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        fitness = [10, 5, 15, 8]  # Lower is better
        
        selector = RouletteWheelSelection()
        parents = selector.select(population, fitness, 2)
        
        assert len(parents) == 2
        assert parents.shape == (2, 2)
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        population = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        fitness = [10, 5, 15, 8]  # Lower is better
        
        selector = TournamentSelection(tournament_size=2)
        parents = selector.select(population, fitness, 2)
        
        assert len(parents) == 2
        assert parents.shape == (2, 2)
    
    def test_selection_with_equal_fitness(self):
        """Test selection with equal fitness values."""
        population = np.array([[1, 2], [3, 4], [5, 6]])
        fitness = [5, 5, 5]  # All equal
        
        selector = RouletteWheelSelection()
        parents = selector.select(population, fitness, 2)
        
        assert len(parents) == 2


class TestCrossoverOperators:
    """Test crossover operators."""
    
    def test_single_point_crossover(self):
        """Test single point crossover."""
        parent1 = np.array([1, 2, 3, 4, 5])
        parent2 = np.array([6, 7, 8, 9, 10])
        
        crossover = SinglePointCrossover()
        child1, child2 = crossover.crossover(parent1, parent2)
        
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)
        
        # Check that some genes are exchanged
        assert not np.array_equal(child1, parent1) or not np.array_equal(child2, parent2)
    
    def test_uniform_crossover(self):
        """Test uniform crossover."""
        parent1 = np.array([1, 2, 3, 4, 5])
        parent2 = np.array([6, 7, 8, 9, 10])
        
        crossover = UniformCrossover(probability=0.5)
        child1, child2 = crossover.crossover(parent1, parent2)
        
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)
    
    def test_arithmetic_crossover(self):
        """Test arithmetic crossover."""
        parent1 = np.array([1.0, 2.0, 3.0])
        parent2 = np.array([4.0, 5.0, 6.0])
        
        crossover = ArithmeticCrossover(alpha=0.5)
        child1, child2 = crossover.crossover(parent1, parent2)
        
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)
        
        # Check arithmetic combination
        expected_child1 = 0.5 * parent1 + 0.5 * parent2
        expected_child2 = 0.5 * parent1 + 0.5 * parent2
        
        np.testing.assert_array_almost_equal(child1, expected_child1)
        np.testing.assert_array_almost_equal(child2, expected_child2)


class TestMutationOperators:
    """Test mutation operators."""
    
    def test_gaussian_mutation(self):
        """Test Gaussian mutation."""
        individual = np.array([1.0, 2.0, 3.0])
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        
        mutation = GaussianMutation(sigma=0.1)
        mutated = mutation.mutate(individual, bounds)
        
        assert len(mutated) == len(individual)
        assert all(bounds[i][0] <= mutated[i] <= bounds[i][1] for i in range(len(mutated)))
    
    def test_uniform_mutation(self):
        """Test uniform mutation."""
        individual = np.array([1.0, 2.0, 3.0])
        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        
        mutation = UniformMutation(mutation_strength=0.1)
        mutated = mutation.mutate(individual, bounds)
        
        assert len(mutated) == len(individual)
        assert all(bounds[i][0] <= mutated[i] <= bounds[i][1] for i in range(len(mutated)))
    
    def test_mutation_bounds_clipping(self):
        """Test that mutation respects bounds."""
        individual = np.array([4.9, -4.9])  # Near bounds
        bounds = [(-5, 5), (-5, 5)]
        
        mutation = GaussianMutation(sigma=1.0)  # Large sigma to force clipping
        
        for _ in range(10):  # Multiple tests due to randomness
            mutated = mutation.mutate(individual, bounds)
            assert all(bounds[i][0] <= mutated[i] <= bounds[i][1] for i in range(len(mutated)))


class TestTestFunctions:
    """Test benchmark functions."""
    
    def test_sphere_function(self):
        """Test sphere function."""
        # Test at origin (global minimum)
        result = TestFunctions.sphere([0, 0, 0])
        assert result == 0.0
        
        # Test with non-zero values
        result = TestFunctions.sphere([1, 1, 1])
        assert result == 3.0
    
    def test_rastrigin_function(self):
        """Test Rastrigin function."""
        # Test at origin (global minimum)
        result = TestFunctions.rastrigin([0, 0])
        assert result == 0.0
        
        # Test with non-zero values
        result = TestFunctions.rastrigin([1, 1])
        assert result > 0
    
    def test_rosenbrock_function(self):
        """Test Rosenbrock function."""
        # Test at global minimum
        result = TestFunctions.rosenbrock([1, 1])
        assert abs(result) < 1e-10
        
        # Test with other values
        result = TestFunctions.rosenbrock([0, 0])
        assert result > 0
    
    def test_function_factory(self):
        """Test function factory."""
        func = get_test_function('sphere')
        assert func is not None
        assert func([0, 0]) == 0.0
        
        func = get_test_function('nonexistent')
        assert func is None
    
    def test_function_info(self):
        """Test function information retrieval."""
        info = TestFunctions.get_function_info('sphere')
        assert 'global_min' in info
        assert 'search_domain' in info
        assert info['global_min'] == 0.0


class TestGeneticAlgorithm:
    """Test genetic algorithm."""
    
    def setup_method(self):
        """Set up test GA."""
        self.fitness_function = TestFunctions.sphere
        self.dimensions = 3
        self.bounds = [(-5, 5)] * self.dimensions
        self.population_size = 20
        
        self.ga = GeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size
        )
    
    def test_initialization(self):
        """Test GA initialization."""
        assert self.ga.population.shape == (self.population_size, self.dimensions)
        assert self.ga.generation == 0
        assert self.ga.best_fitness == float('inf')
    
    def test_population_bounds(self):
        """Test that initial population respects bounds."""
        for individual in self.ga.population:
            for i, (min_val, max_val) in enumerate(self.bounds):
                assert min_val <= individual[i] <= max_val
    
    def test_evaluate_population(self):
        """Test population evaluation."""
        fitness_values = self.ga.evaluate_population()
        
        assert len(fitness_values) == self.population_size
        assert all(f >= 0 for f in fitness_values)  # Sphere function is non-negative
    
    def test_evolve_generation(self):
        """Test generation evolution."""
        initial_generation = self.ga.generation
        self.ga.evolve_generation()
        
        assert self.ga.generation == initial_generation + 1
        assert len(self.ga.fitness_values) == self.population_size
        assert self.ga.best_individual is not None
        assert self.ga.best_fitness < float('inf')
    
    def test_run_algorithm(self):
        """Test running the algorithm."""
        result = self.ga.run(max_generations=10)
        
        assert 'best_individual' in result
        assert 'best_fitness' in result
        assert 'generations_run' in result
        assert result['generations_run'] <= 10
        assert self.ga.generation > 0
    
    def test_convergence_detection(self):
        """Test convergence detection."""
        # Run algorithm with tight convergence criteria
        result = self.ga.run(
            max_generations=100,
            convergence_threshold=1e-6,
            convergence_generations=5
        )
        
        # Should terminate due to convergence or max generations
        assert result['generations_run'] <= 100
    
    def test_target_fitness(self):
        """Test target fitness termination."""
        result = self.ga.run(
            max_generations=100,
            target_fitness=0.1  # Easy target for sphere function
        )
        
        # Should find solution close to target
        assert result['best_fitness'] <= 0.1 or result['generations_run'] == 100
    
    def test_statistics(self):
        """Test statistics collection."""
        self.ga.evolve_generation()
        stats = self.ga.get_statistics()
        
        assert 'generation' in stats
        assert 'best_fitness' in stats
        assert 'avg_fitness' in stats
        assert 'std_fitness' in stats
        assert 'population_size' in stats
    
    def test_invalid_bounds(self):
        """Test error handling for invalid bounds."""
        with pytest.raises(ValueError):
            # Invalid bounds (min > max)
            invalid_bounds = [(5, -5), (0, 1)]
            GeneticAlgorithm(
                fitness_function=self.fitness_function,
                dimensions=2,
                bounds=invalid_bounds,
                population_size=10
            )
        
        with pytest.raises(ValueError):
            # Mismatched dimensions and bounds
            mismatched_bounds = [(0, 1)]  # Only 1 bound for 3D problem
            GeneticAlgorithm(
                fitness_function=self.fitness_function,
                dimensions=3,
                bounds=mismatched_bounds,
                population_size=10
            )


if __name__ == '__main__':
    pytest.main([__file__])