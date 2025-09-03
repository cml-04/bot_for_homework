"""
Tests for distributed genetic algorithm components.
"""

import pytest
import numpy as np
import ray
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distributed_ga import DistributedGeneticAlgorithm, FitnessEvaluator, GeneticOperator
from test_functions import TestFunctions
from genetic_algorithm import RouletteWheelSelection, SinglePointCrossover, GaussianMutation


@pytest.fixture(scope="session")
def ray_context():
    """Initialize Ray for testing."""
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestFitnessEvaluator:
    """Test remote fitness evaluator."""
    
    def test_single_evaluation(self, ray_context):
        """Test single individual evaluation."""
        evaluator = FitnessEvaluator.remote(TestFunctions.sphere)
        
        individual = np.array([1.0, 2.0, 3.0])
        fitness = ray.get(evaluator.evaluate_single.remote(individual))
        
        expected = TestFunctions.sphere(individual)
        assert fitness == expected
    
    def test_batch_evaluation(self, ray_context):
        """Test batch evaluation."""
        evaluator = FitnessEvaluator.remote(TestFunctions.sphere)
        
        individuals = [
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.array([3.0, 4.0])
        ]
        
        fitness_values = ray.get(evaluator.evaluate_batch.remote(individuals))
        
        assert len(fitness_values) == 3
        assert fitness_values[0] == 5.0  # 1^2 + 2^2
        assert fitness_values[1] == 0.0  # 0^2 + 0^2
        assert fitness_values[2] == 25.0  # 3^2 + 4^2
    
    def test_evaluator_stats(self, ray_context):
        """Test evaluator statistics tracking."""
        evaluator = FitnessEvaluator.remote(TestFunctions.sphere)
        
        # Perform some evaluations
        ray.get(evaluator.evaluate_single.remote(np.array([1.0, 2.0])))
        ray.get(evaluator.evaluate_batch.remote([np.array([0.0, 0.0])]))
        
        stats = ray.get(evaluator.get_stats.remote())
        
        assert 'evaluations_count' in stats
        assert stats['evaluations_count'] == 2
        assert 'node_id' in stats
        assert 'worker_id' in stats


class TestGeneticOperator:
    """Test remote genetic operator."""
    
    def setup_method(self):
        """Set up test data."""
        self.population = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        self.fitness = [10.0, 5.0, 15.0, 8.0]
        self.bounds = [(-10, 10), (-10, 10)]
    
    def test_offspring_generation(self, ray_context):
        """Test offspring generation."""
        operator = GeneticOperator.remote(
            RouletteWheelSelection(),
            SinglePointCrossover(),
            GaussianMutation(),
            crossover_probability=0.8,
            mutation_probability=0.1
        )
        
        offspring = ray.get(operator.generate_offspring.remote(
            self.population, self.fitness, self.bounds, 4
        ))
        
        assert offspring.shape == (4, 2)
        
        # Check bounds
        for individual in offspring:
            for i, (min_val, max_val) in enumerate(self.bounds):
                assert min_val <= individual[i] <= max_val
    
    def test_operator_stats(self, ray_context):
        """Test operator statistics tracking."""
        operator = GeneticOperator.remote(
            RouletteWheelSelection(),
            SinglePointCrossover(),
            GaussianMutation()
        )
        
        # Generate some offspring
        ray.get(operator.generate_offspring.remote(
            self.population, self.fitness, self.bounds, 2
        ))
        
        stats = ray.get(operator.get_stats.remote())
        
        assert 'operations_count' in stats
        assert stats['operations_count'] == 2
        assert 'node_id' in stats
        assert 'worker_id' in stats


class TestDistributedGeneticAlgorithm:
    """Test distributed genetic algorithm."""
    
    def setup_method(self):
        """Set up test DGA."""
        self.fitness_function = TestFunctions.sphere
        self.dimensions = 3
        self.bounds = [(-5, 5)] * self.dimensions
        self.population_size = 20
    
    def test_initialization(self, ray_context):
        """Test DGA initialization."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2
        )
        
        assert dga.population.shape == (self.population_size, self.dimensions)
        assert len(dga.fitness_evaluators) == 2
        assert len(dga.genetic_operators) == 2
        assert dga.generation == 0
        
        dga.shutdown()
    
    def test_parallel_evaluation(self, ray_context):
        """Test parallel fitness evaluation."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2
        )
        
        fitness_values = dga.evaluate_population_parallel(dga.population)
        
        assert len(fitness_values) == self.population_size
        assert all(f >= 0 for f in fitness_values)  # Sphere function is non-negative
        
        dga.shutdown()
    
    def test_parallel_offspring_generation(self, ray_context):
        """Test parallel offspring generation."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2
        )
        
        # Evaluate population first
        fitness_values = dga.evaluate_population_parallel(dga.population)
        
        # Generate offspring
        offspring = dga.generate_offspring_parallel(
            dga.population, fitness_values, 10
        )
        
        assert offspring.shape == (10, self.dimensions)
        
        # Check bounds
        for individual in offspring:
            for i, (min_val, max_val) in enumerate(self.bounds):
                assert min_val <= individual[i] <= max_val
        
        dga.shutdown()
    
    def test_generation_evolution(self, ray_context):
        """Test evolution of a single generation."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2
        )
        
        initial_generation = dga.generation
        dga.evolve_generation()
        
        assert dga.generation == initial_generation + 1
        assert len(dga.fitness_values) == self.population_size
        assert dga.best_individual is not None
        assert dga.best_fitness < float('inf')
        
        dga.shutdown()
    
    def test_run_algorithm(self, ray_context):
        """Test running the distributed algorithm."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2
        )
        
        result = dga.run(max_generations=5)
        
        assert 'best_individual' in result
        assert 'best_fitness' in result
        assert 'generations_run' in result
        assert 'total_time' in result
        assert 'worker_statistics' in result
        assert result['generations_run'] <= 5
        
        # Check worker statistics
        worker_stats = result['worker_statistics']
        assert 'evaluator_stats' in worker_stats
        assert 'operator_stats' in worker_stats
        assert 'total_evaluations' in worker_stats
        assert 'total_operations' in worker_stats
        
        dga.shutdown()
    
    def test_convergence_detection(self, ray_context):
        """Test convergence detection in distributed mode."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=2,  # Smaller problem for faster convergence
            bounds=[(-1, 1)] * 2,
            population_size=20,
            num_workers=2
        )
        
        result = dga.run(
            max_generations=50,
            convergence_threshold=1e-6,
            convergence_generations=5
        )
        
        # Should terminate due to convergence or max generations
        assert result['generations_run'] <= 50
        
        dga.shutdown()
    
    def test_worker_statistics(self, ray_context):
        """Test worker statistics collection."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2
        )
        
        # Run a few generations
        dga.run(max_generations=3)
        
        # Get statistics
        stats = dga.get_statistics()
        
        assert 'generation' in stats
        assert 'best_fitness' in stats
        assert 'num_workers' in stats
        assert 'batch_size' in stats
        assert stats['num_workers'] == 2
        
        dga.shutdown()
    
    def test_batch_size_calculation(self, ray_context):
        """Test automatic batch size calculation."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=100,
            num_workers=4
        )
        
        # Batch size should be reasonable
        assert dga.batch_size > 0
        assert dga.batch_size <= dga.population_size
        
        dga.shutdown()
    
    def test_custom_batch_size(self, ray_context):
        """Test custom batch size setting."""
        custom_batch_size = 10
        
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2,
            batch_size=custom_batch_size
        )
        
        assert dga.batch_size == custom_batch_size
        
        dga.shutdown()
    
    def test_elitism_preservation(self, ray_context):
        """Test that elitism preserves best individuals."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=self.fitness_function,
            dimensions=self.dimensions,
            bounds=self.bounds,
            population_size=self.population_size,
            num_workers=2,
            elitism_size=2
        )
        
        # Run one generation to get initial best fitness
        dga.evolve_generation()
        initial_best = dga.best_fitness
        
        # Run another generation
        dga.evolve_generation()
        
        # Best fitness should not get worse (minimization)
        assert dga.best_fitness <= initial_best
        
        dga.shutdown()


class TestDistributedPerformance:
    """Test performance aspects of distributed GA."""
    
    @pytest.mark.slow
    def test_speedup_with_workers(self, ray_context):
        """Test that adding workers improves performance."""
        problem_config = {
            'fitness_function': TestFunctions.rastrigin,
            'dimensions': 5,
            'bounds': [(-5, 5)] * 5,
            'population_size': 100
        }
        
        # Single worker
        dga1 = DistributedGeneticAlgorithm(num_workers=1, **problem_config)
        start_time = time.time()
        result1 = dga1.run(max_generations=10)
        time1 = time.time() - start_time
        dga1.shutdown()
        
        # Multiple workers
        dga2 = DistributedGeneticAlgorithm(num_workers=2, **problem_config)
        start_time = time.time()
        result2 = dga2.run(max_generations=10)
        time2 = time.time() - start_time
        dga2.shutdown()
        
        # Multiple workers should be faster (or at least not significantly slower)
        # Allow some variance due to overhead
        speedup_ratio = time1 / time2
        assert speedup_ratio > 0.5  # At least not more than 2x slower
        
        # Both should find reasonable solutions
        assert result1['best_fitness'] < 1000  # Reasonable for Rastrigin
        assert result2['best_fitness'] < 1000
    
    def test_load_balancing(self, ray_context):
        """Test that work is distributed among workers."""
        dga = DistributedGeneticAlgorithm(
            fitness_function=TestFunctions.sphere,
            dimensions=3,
            bounds=[(-5, 5)] * 3,
            population_size=100,
            num_workers=3
        )
        
        result = dga.run(max_generations=5)
        
        # Check that all workers did some work
        worker_stats = result['worker_statistics']
        evaluator_stats = worker_stats['evaluator_stats']
        operator_stats = worker_stats['operator_stats']
        
        # All evaluators should have done some evaluations
        eval_counts = [stat['evaluations_count'] for stat in evaluator_stats]
        assert all(count > 0 for count in eval_counts)
        
        # All operators should have done some operations
        op_counts = [stat['operations_count'] for stat in operator_stats]
        assert all(count > 0 for count in op_counts)
        
        # Work should be reasonably balanced (no worker does more than 80% of total work)
        total_evals = sum(eval_counts)
        total_ops = sum(op_counts)
        
        for count in eval_counts:
            assert count / total_evals < 0.8
        
        for count in op_counts:
            assert count / total_ops < 0.8
        
        dga.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])