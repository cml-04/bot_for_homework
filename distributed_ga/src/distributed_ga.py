"""
Distributed genetic algorithm implementation using Ray framework.
"""

import ray
import numpy as np
import time
import logging
from typing import List, Tuple, Callable, Optional, Dict, Any, Union
from concurrent.futures import as_completed
import psutil

from genetic_algorithm import (
    GeneticAlgorithm, SelectionOperator, CrossoverOperator, MutationOperator,
    RouletteWheelSelection, SinglePointCrossover, GaussianMutation
)
from utils import setup_logging, get_system_info, Timer, Statistics


@ray.remote
class FitnessEvaluator:
    """Remote actor for fitness evaluation."""
    
    def __init__(self, fitness_function: Callable[[np.ndarray], float]):
        self.fitness_function = fitness_function
        self.evaluations_count = 0
    
    def evaluate_batch(self, individuals: List[np.ndarray]) -> List[float]:
        """Evaluate a batch of individuals."""
        fitness_values = []
        for individual in individuals:
            fitness = self.fitness_function(individual)
            fitness_values.append(fitness)
            self.evaluations_count += 1
        return fitness_values
    
    def evaluate_single(self, individual: np.ndarray) -> float:
        """Evaluate a single individual."""
        self.evaluations_count += 1
        return self.fitness_function(individual)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        return {
            'evaluations_count': self.evaluations_count,
            'node_id': ray.get_runtime_context().node_id.hex(),
            'worker_id': ray.get_runtime_context().worker_id.hex()
        }


@ray.remote
class GeneticOperator:
    """Remote actor for genetic operations."""
    
    def __init__(self, 
                 selection_operator: SelectionOperator,
                 crossover_operator: CrossoverOperator,
                 mutation_operator: MutationOperator,
                 crossover_probability: float = 0.8,
                 mutation_probability: float = 0.1):
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.operations_count = 0
    
    def generate_offspring(self, population: np.ndarray, fitness: List[float],
                          bounds: List[Tuple[float, float]], num_offspring: int) -> np.ndarray:
        """Generate offspring from population."""
        offspring = []
        
        while len(offspring) < num_offspring:
            # Selection
            parents = self.selection_operator.select(population, fitness, 2)
            parent1, parent2 = parents[0], parents[1]
            
            # Crossover
            if np.random.random() < self.crossover_probability:
                child1, child2 = self.crossover_operator.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.mutation_probability:
                child1 = self.mutation_operator.mutate(child1, bounds)
            
            if np.random.random() < self.mutation_probability:
                child2 = self.mutation_operator.mutate(child2, bounds)
            
            offspring.append(child1)
            if len(offspring) < num_offspring:
                offspring.append(child2)
            
            self.operations_count += 2
        
        return np.array(offspring[:num_offspring])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get operator statistics."""
        return {
            'operations_count': self.operations_count,
            'node_id': ray.get_runtime_context().node_id.hex(),
            'worker_id': ray.get_runtime_context().worker_id.hex()
        }


class DistributedGeneticAlgorithm:
    """Distributed genetic algorithm using Ray."""
    
    def __init__(self,
                 fitness_function: Callable[[np.ndarray], float],
                 dimensions: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 100,
                 selection_operator: Optional[SelectionOperator] = None,
                 crossover_operator: Optional[CrossoverOperator] = None,
                 mutation_operator: Optional[MutationOperator] = None,
                 crossover_probability: float = 0.8,
                 mutation_probability: float = 0.1,
                 elitism_size: int = 2,
                 num_workers: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 island_model: bool = False,
                 migration_rate: float = 0.1,
                 migration_interval: int = 10):
        """
        Initialize distributed genetic algorithm.
        
        Args:
            fitness_function: Function to evaluate fitness
            dimensions: Number of dimensions
            bounds: List of (min, max) tuples for each dimension
            population_size: Size of population
            selection_operator: Selection operator
            crossover_operator: Crossover operator
            mutation_operator: Mutation operator
            crossover_probability: Probability of crossover
            mutation_probability: Probability of mutation
            elitism_size: Number of elite individuals to preserve
            num_workers: Number of worker nodes (auto-detect if None)
            batch_size: Batch size for parallel evaluation
            island_model: Whether to use island model
            migration_rate: Rate of migration between islands
            migration_interval: Interval between migrations
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.fitness_function = fitness_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_size = elitism_size
        self.island_model = island_model
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        
        # Set default operators
        self.selection_operator = selection_operator or RouletteWheelSelection()
        self.crossover_operator = crossover_operator or SinglePointCrossover()
        self.mutation_operator = mutation_operator or GaussianMutation()
        
        # Determine number of workers
        if num_workers is None:
            self.num_workers = min(ray.cluster_resources().get('CPU', 1), psutil.cpu_count())
        else:
            self.num_workers = num_workers
        
        # Set batch size
        if batch_size is None:
            self.batch_size = max(1, population_size // (self.num_workers * 4))
        else:
            self.batch_size = batch_size
        
        # Initialize remote actors
        self._initialize_actors()
        
        # Initialize population and statistics
        self.population = self._initialize_population()
        self.fitness_values = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness = float('inf')
        self.statistics = Statistics()
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info(f"Initialized distributed GA with {self.num_workers} workers")
    
    def _initialize_actors(self) -> None:
        """Initialize Ray actors."""
        # Create fitness evaluators
        self.fitness_evaluators = [
            FitnessEvaluator.remote(self.fitness_function) 
            for _ in range(self.num_workers)
        ]
        
        # Create genetic operators
        self.genetic_operators = [
            GeneticOperator.remote(
                self.selection_operator,
                self.crossover_operator,
                self.mutation_operator,
                self.crossover_probability,
                self.mutation_probability
            ) for _ in range(self.num_workers)
        ]
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population."""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for min_val, max_val in self.bounds:
                gene = np.random.uniform(min_val, max_val)
                individual.append(gene)
            population.append(individual)
        
        return np.array(population)
    
    def evaluate_population_parallel(self, population: np.ndarray) -> List[float]:
        """Evaluate population in parallel."""
        # Split population into batches
        batches = [population[i:i + self.batch_size] 
                  for i in range(0, len(population), self.batch_size)]
        
        # Distribute batches to workers
        futures = []
        for i, batch in enumerate(batches):
            evaluator = self.fitness_evaluators[i % len(self.fitness_evaluators)]
            future = evaluator.evaluate_batch.remote(batch.tolist())
            futures.append(future)
        
        # Collect results
        fitness_values = []
        for future in futures:
            batch_fitness = ray.get(future)
            fitness_values.extend(batch_fitness)
        
        return fitness_values
    
    def generate_offspring_parallel(self, population: np.ndarray, 
                                  fitness: List[float], num_offspring: int) -> np.ndarray:
        """Generate offspring in parallel."""
        # Calculate offspring per worker
        offspring_per_worker = num_offspring // len(self.genetic_operators)
        remaining_offspring = num_offspring % len(self.genetic_operators)
        
        # Distribute work to operators
        futures = []
        for i, operator in enumerate(self.genetic_operators):
            worker_offspring = offspring_per_worker
            if i < remaining_offspring:
                worker_offspring += 1
            
            if worker_offspring > 0:
                future = operator.generate_offspring.remote(
                    population, fitness, self.bounds, worker_offspring
                )
                futures.append(future)
        
        # Collect offspring
        all_offspring = []
        for future in futures:
            offspring_batch = ray.get(future)
            all_offspring.extend(offspring_batch)
        
        return np.array(all_offspring[:num_offspring])
    
    def evolve_generation(self) -> None:
        """Evolve one generation using distributed computing."""
        with Timer("Generation Evolution") as timer:
            # Evaluate population in parallel
            self.fitness_values = self.evaluate_population_parallel(self.population)
            
            # Update best individual
            min_fitness_idx = np.argmin(self.fitness_values)
            if self.fitness_values[min_fitness_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[min_fitness_idx]
                self.best_individual = self.population[min_fitness_idx].copy()
            
            # Calculate diversity
            from utils import calculate_diversity
            diversity = calculate_diversity(self.population)
            
            # Record statistics
            self.statistics.add_generation_stats(
                self.generation,
                self.best_fitness,
                np.mean(self.fitness_values),
                diversity
            )
            
            # Create new population
            new_population = []
            
            # Elitism - preserve best individuals
            if self.elitism_size > 0:
                elite_indices = np.argsort(self.fitness_values)[:self.elitism_size]
                for idx in elite_indices:
                    new_population.append(self.population[idx].copy())
            
            # Generate offspring in parallel
            num_offspring = self.population_size - len(new_population)
            if num_offspring > 0:
                offspring = self.generate_offspring_parallel(
                    self.population, self.fitness_values, num_offspring
                )
                new_population.extend(offspring)
            
            # Update population
            self.population = np.array(new_population[:self.population_size])
            self.generation += 1
            
            # Log progress
            self.logger.info(
                f"Generation {self.generation}: Best={self.best_fitness:.6f}, "
                f"Avg={np.mean(self.fitness_values):.6f}, "
                f"Time={timer.elapsed_time:.2f}s"
            )
    
    def run(self, max_generations: int, target_fitness: Optional[float] = None,
            convergence_threshold: float = 1e-6, convergence_generations: int = 50,
            checkpoint_interval: Optional[int] = None, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the distributed genetic algorithm.
        
        Args:
            max_generations: Maximum number of generations
            target_fitness: Target fitness value (optional)
            convergence_threshold: Threshold for convergence detection
            convergence_generations: Number of generations to check for convergence
            checkpoint_interval: Interval for saving checkpoints
            checkpoint_path: Path for saving checkpoints
            
        Returns:
            Result dictionary
        """
        start_time = time.time()
        start_generation = self.generation
        
        self.logger.info(f"Starting optimization for {max_generations} generations")
        
        try:
            for gen in range(max_generations):
                self.evolve_generation()
                
                # Check target fitness
                if target_fitness is not None and self.best_fitness <= target_fitness:
                    self.logger.info(f"Target fitness {target_fitness} reached!")
                    break
                
                # Check convergence
                if self._check_convergence(convergence_threshold, convergence_generations):
                    self.logger.info("Algorithm converged!")
                    break
                
                # Save checkpoint
                if (checkpoint_interval and checkpoint_path and 
                    (gen + 1) % checkpoint_interval == 0):
                    self._save_checkpoint(checkpoint_path)
                
                # Migration for island model
                if self.island_model and (gen + 1) % self.migration_interval == 0:
                    self._perform_migration()
        
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get worker statistics
        worker_stats = self._get_worker_statistics()
        
        result = {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'generations_run': self.generation - start_generation,
            'total_generations': self.generation,
            'total_time': total_time,
            'convergence_data': self.statistics.get_convergence_data(),
            'final_population': self.population,
            'final_fitness': self.fitness_values,
            'worker_statistics': worker_stats,
            'system_info': get_system_info()
        }
        
        self.logger.info(
            f"Optimization completed: Best fitness = {self.best_fitness:.6f}, "
            f"Time = {total_time:.2f}s"
        )
        
        return result
    
    def _check_convergence(self, threshold: float, num_generations: int) -> bool:
        """Check if algorithm has converged."""
        if len(self.statistics.history) < num_generations:
            return False
        
        recent_fitness = [entry['best_fitness'] for entry in self.statistics.history[-num_generations:]]
        fitness_range = max(recent_fitness) - min(recent_fitness)
        return fitness_range < threshold
    
    def _save_checkpoint(self, checkpoint_path: str) -> None:
        """Save algorithm state to checkpoint."""
        from utils import create_checkpoint
        
        config = {
            'population_size': self.population_size,
            'dimensions': self.dimensions,
            'bounds': self.bounds,
            'crossover_probability': self.crossover_probability,
            'mutation_probability': self.mutation_probability,
            'elitism_size': self.elitism_size
        }
        
        create_checkpoint(
            self.population, self.fitness_values, 
            self.generation, config, checkpoint_path
        )
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _perform_migration(self) -> None:
        """Perform migration for island model."""
        # Simple migration strategy: exchange best individuals
        if self.num_workers > 1:
            # This is a simplified implementation
            # In a real island model, we would have separate populations per island
            num_migrants = max(1, int(self.population_size * self.migration_rate))
            
            # Get best individuals
            best_indices = np.argsort(self.fitness_values)[:num_migrants]
            
            # Replace worst individuals with mutated copies of best
            worst_indices = np.argsort(self.fitness_values)[-num_migrants:]
            
            for i, (best_idx, worst_idx) in enumerate(zip(best_indices, worst_indices)):
                mutated = self.mutation_operator.mutate(
                    self.population[best_idx], self.bounds
                )
                self.population[worst_idx] = mutated
            
            self.logger.info(f"Migration performed: {num_migrants} individuals")
    
    def _get_worker_statistics(self) -> Dict[str, Any]:
        """Get statistics from all workers."""
        evaluator_futures = [evaluator.get_stats.remote() for evaluator in self.fitness_evaluators]
        operator_futures = [operator.get_stats.remote() for operator in self.genetic_operators]
        
        evaluator_stats = ray.get(evaluator_futures)
        operator_stats = ray.get(operator_futures)
        
        return {
            'evaluator_stats': evaluator_stats,
            'operator_stats': operator_stats,
            'total_evaluations': sum(stat['evaluations_count'] for stat in evaluator_stats),
            'total_operations': sum(stat['operations_count'] for stat in operator_stats)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current algorithm statistics."""
        latest_stats = self.statistics.get_latest_stats()
        latest_stats.update({
            'population_size': len(self.population),
            'num_workers': self.num_workers,
            'batch_size': self.batch_size
        })
        return latest_stats
    
    def shutdown(self) -> None:
        """Shutdown the distributed algorithm."""
        # Clean up Ray actors
        for evaluator in self.fitness_evaluators:
            ray.kill(evaluator)
        
        for operator in self.genetic_operators:
            ray.kill(operator)
        
        self.logger.info("Distributed GA shutdown completed")


class MasterWorkerGA:
    """Master-worker distributed genetic algorithm for cross-platform deployment."""
    
    def __init__(self, master_config: Dict[str, Any], worker_configs: List[Dict[str, Any]]):
        """
        Initialize master-worker GA.
        
        Args:
            master_config: Configuration for master node
            worker_configs: List of configurations for worker nodes
        """
        self.master_config = master_config
        self.worker_configs = worker_configs
        self.logger = setup_logging()
        
        # Initialize Ray cluster
        self._initialize_cluster()
    
    def _initialize_cluster(self) -> None:
        """Initialize Ray cluster with master and workers."""
        # This would be implemented for actual cluster setup
        # For now, we use local Ray initialization
        if not ray.is_initialized():
            ray.init(address=self.master_config.get('ray_address', None))
        
        self.logger.info("Ray cluster initialized")
    
    def run_optimization(self, problem_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimization problem on the cluster."""
        # Create distributed GA
        from test_functions import get_test_function
        
        fitness_func = get_test_function(problem_config['function_name'])
        if fitness_func is None:
            raise ValueError(f"Unknown function: {problem_config['function_name']}")
        
        dga = DistributedGeneticAlgorithm(
            fitness_function=fitness_func,
            dimensions=problem_config['dimensions'],
            bounds=problem_config['bounds'],
            population_size=problem_config.get('population_size', 100),
            num_workers=problem_config.get('num_workers', None)
        )
        
        # Run optimization
        result = dga.run(
            max_generations=problem_config.get('max_generations', 100),
            target_fitness=problem_config.get('target_fitness', None)
        )
        
        # Cleanup
        dga.shutdown()
        
        return result