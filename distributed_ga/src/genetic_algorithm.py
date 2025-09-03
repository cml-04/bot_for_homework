"""
Core genetic algorithm implementation.
"""

import numpy as np
import random
from typing import List, Tuple, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod
from utils import normalize_fitness, clip_to_bounds, validate_bounds


class SelectionOperator(ABC):
    """Abstract base class for selection operators."""
    
    @abstractmethod
    def select(self, population: np.ndarray, fitness: List[float], 
              num_parents: int) -> np.ndarray:
        """Select parents from population."""
        pass


class RouletteWheelSelection(SelectionOperator):
    """Roulette wheel selection operator."""
    
    def select(self, population: np.ndarray, fitness: List[float], 
              num_parents: int) -> np.ndarray:
        """
        Select parents using roulette wheel selection.
        
        Args:
            population: Population array
            fitness: Fitness values (minimization - lower is better)
            num_parents: Number of parents to select
            
        Returns:
            Selected parents
        """
        # Convert to maximization problem for roulette wheel
        # Add small constant to avoid division by zero
        max_fitness = max(fitness) + 1e-10
        inv_fitness = [max_fitness - f for f in fitness]
        
        # Normalize to get probabilities
        total_fitness = sum(inv_fitness)
        if total_fitness == 0:
            # If all fitness values are the same, select randomly
            indices = np.random.choice(len(population), num_parents, replace=True)
            return population[indices]
        
        probabilities = [f / total_fitness for f in inv_fitness]
        
        # Select parents
        selected_indices = np.random.choice(
            len(population), 
            size=num_parents, 
            replace=True, 
            p=probabilities
        )
        
        return population[selected_indices]


class TournamentSelection(SelectionOperator):
    """Tournament selection operator."""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: np.ndarray, fitness: List[float], 
              num_parents: int) -> np.ndarray:
        """
        Select parents using tournament selection.
        
        Args:
            population: Population array
            fitness: Fitness values (minimization)
            num_parents: Number of parents to select
            
        Returns:
            Selected parents
        """
        selected = []
        
        for _ in range(num_parents):
            # Select tournament participants
            tournament_indices = np.random.choice(
                len(population), 
                size=min(self.tournament_size, len(population)), 
                replace=False
            )
            
            # Find best individual in tournament (lowest fitness)
            best_idx = tournament_indices[0]
            best_fitness = fitness[best_idx]
            
            for idx in tournament_indices[1:]:
                if fitness[idx] < best_fitness:
                    best_idx = idx
                    best_fitness = fitness[idx]
            
            selected.append(population[best_idx])
        
        return np.array(selected)


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""
    
    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        pass


class SinglePointCrossover(CrossoverOperator):
    """Single point crossover operator."""
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        length = len(parent1)
        if length <= 1:
            return parent1.copy(), parent2.copy()
        
        # Choose crossover point
        crossover_point = random.randint(1, length - 1)
        
        # Create offspring
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return offspring1, offspring2


class UniformCrossover(CrossoverOperator):
    """Uniform crossover operator."""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        for i in range(len(parent1)):
            if random.random() < self.probability:
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
        
        return offspring1, offspring2


class ArithmeticCrossover(CrossoverOperator):
    """Arithmetic crossover operator."""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform arithmetic crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        offspring1 = self.alpha * parent1 + (1 - self.alpha) * parent2
        offspring2 = (1 - self.alpha) * parent1 + self.alpha * parent2
        
        return offspring1, offspring2


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""
    
    @abstractmethod
    def mutate(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Mutate an individual."""
        pass


class GaussianMutation(MutationOperator):
    """Gaussian mutation operator."""
    
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma
    
    def mutate(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Perform Gaussian mutation.
        
        Args:
            individual: Individual to mutate
            bounds: Bounds for each dimension
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(len(individual)):
            # Add Gaussian noise
            noise = np.random.normal(0, self.sigma)
            mutated[i] += noise
        
        # Clip to bounds
        return clip_to_bounds(mutated, bounds)


class UniformMutation(MutationOperator):
    """Uniform mutation operator."""
    
    def __init__(self, mutation_strength: float = 0.1):
        self.mutation_strength = mutation_strength
    
    def mutate(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Perform uniform mutation.
        
        Args:
            individual: Individual to mutate
            bounds: Bounds for each dimension
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for i in range(len(individual)):
            min_val, max_val = bounds[i]
            range_val = max_val - min_val
            
            # Add uniform noise
            noise = np.random.uniform(-self.mutation_strength, self.mutation_strength) * range_val
            mutated[i] += noise
        
        # Clip to bounds
        return clip_to_bounds(mutated, bounds)


class GeneticAlgorithm:
    """Core genetic algorithm implementation."""
    
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
                 elitism_size: int = 2):
        """
        Initialize genetic algorithm.
        
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
        """
        self.fitness_function = fitness_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_size = elitism_size
        
        # Validate bounds
        if not validate_bounds(bounds):
            raise ValueError("Invalid bounds specified")
        
        if len(bounds) != dimensions:
            raise ValueError("Number of bounds must match dimensions")
        
        # Set default operators
        self.selection_operator = selection_operator or RouletteWheelSelection()
        self.crossover_operator = crossover_operator or SinglePointCrossover()
        self.mutation_operator = mutation_operator or GaussianMutation()
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_values = []
        self.generation = 0
        
        # Statistics
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []
    
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
    
    def evaluate_population(self) -> List[float]:
        """Evaluate fitness of entire population."""
        fitness_values = []
        
        for individual in self.population:
            fitness = self.fitness_function(individual)
            fitness_values.append(fitness)
        
        return fitness_values
    
    def evolve_generation(self) -> None:
        """Evolve one generation."""
        # Evaluate population
        self.fitness_values = self.evaluate_population()
        
        # Update best individual
        min_fitness_idx = np.argmin(self.fitness_values)
        if self.fitness_values[min_fitness_idx] < self.best_fitness:
            self.best_fitness = self.fitness_values[min_fitness_idx]
            self.best_individual = self.population[min_fitness_idx].copy()
        
        # Store fitness history
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean(self.fitness_values),
            'std_fitness': np.std(self.fitness_values)
        })
        
        # Create new population
        new_population = []
        
        # Elitism - preserve best individuals
        if self.elitism_size > 0:
            elite_indices = np.argsort(self.fitness_values)[:self.elitism_size]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parents = self.selection_operator.select(
                self.population, self.fitness_values, 2
            )
            
            parent1, parent2 = parents[0], parents[1]
            
            # Crossover
            if random.random() < self.crossover_probability:
                offspring1, offspring2 = self.crossover_operator.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_probability:
                offspring1 = self.mutation_operator.mutate(offspring1, self.bounds)
            
            if random.random() < self.mutation_probability:
                offspring2 = self.mutation_operator.mutate(offspring2, self.bounds)
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        # Update population
        self.population = np.array(new_population[:self.population_size])
        self.generation += 1
    
    def run(self, max_generations: int, target_fitness: Optional[float] = None,
            convergence_threshold: float = 1e-6, convergence_generations: int = 50) -> Dict[str, Any]:
        """
        Run the genetic algorithm.
        
        Args:
            max_generations: Maximum number of generations
            target_fitness: Target fitness value (optional)
            convergence_threshold: Threshold for convergence detection
            convergence_generations: Number of generations to check for convergence
            
        Returns:
            Result dictionary
        """
        start_generation = self.generation
        
        for _ in range(max_generations):
            self.evolve_generation()
            
            # Check target fitness
            if target_fitness is not None and self.best_fitness <= target_fitness:
                break
            
            # Check convergence
            if self._check_convergence(convergence_threshold, convergence_generations):
                break
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'generations_run': self.generation - start_generation,
            'total_generations': self.generation,
            'fitness_history': self.fitness_history,
            'final_population': self.population,
            'final_fitness': self.fitness_values
        }
    
    def _check_convergence(self, threshold: float, num_generations: int) -> bool:
        """Check if algorithm has converged."""
        if len(self.fitness_history) < num_generations:
            return False
        
        recent_fitness = [entry['best_fitness'] for entry in self.fitness_history[-num_generations:]]
        
        # Check if fitness has not improved significantly
        fitness_range = max(recent_fitness) - min(recent_fitness)
        return fitness_range < threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current algorithm statistics."""
        if not self.fitness_values:
            return {}
        
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean(self.fitness_values),
            'std_fitness': np.std(self.fitness_values),
            'population_size': len(self.population)
        }