"""
Utility functions for the distributed genetic algorithm.
"""

import numpy as np
import time
import logging
import os
import psutil
from typing import List, Tuple, Any, Dict
import yaml


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("distributed_ga")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def generate_population_bounds(dimensions: int, bounds: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Generate bounds for population initialization.
    
    Args:
        dimensions: Number of dimensions
        bounds: Tuple of (min_val, max_val)
        
    Returns:
        List of bounds for each dimension
    """
    return [bounds for _ in range(dimensions)]


def normalize_fitness(fitness_values: List[float]) -> List[float]:
    """
    Normalize fitness values to [0, 1] range.
    
    Args:
        fitness_values: List of fitness values
        
    Returns:
        Normalized fitness values
    """
    if not fitness_values:
        return []
    
    min_fit = min(fitness_values)
    max_fit = max(fitness_values)
    
    if max_fit == min_fit:
        return [1.0] * len(fitness_values)
    
    return [(fit - min_fit) / (max_fit - min_fit) for fit in fitness_values]


def calculate_diversity(population: np.ndarray) -> float:
    """
    Calculate population diversity using average pairwise distance.
    
    Args:
        population: Population array (n_individuals, n_dimensions)
        
    Returns:
        Average diversity measure
    """
    if len(population) < 2:
        return 0.0
    
    total_distance = 0.0
    count = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = np.linalg.norm(population[i] - population[j])
            total_distance += distance
            count += 1
    
    return total_distance / count if count > 0 else 0.0


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for monitoring.
    
    Returns:
        Dictionary with system information
    """
    return {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
    }


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class Statistics:
    """Class for collecting and managing statistics."""
    
    def __init__(self):
        self.data = {}
        self.history = []
    
    def add_generation_stats(self, generation: int, best_fitness: float, 
                           avg_fitness: float, diversity: float) -> None:
        """Add statistics for a generation."""
        stats = {
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'diversity': diversity,
            'timestamp': time.time()
        }
        self.history.append(stats)
    
    def get_convergence_data(self) -> Tuple[List[int], List[float]]:
        """Get convergence data for plotting."""
        generations = [stat['generation'] for stat in self.history]
        best_fitness = [stat['best_fitness'] for stat in self.history]
        return generations, best_fitness
    
    def get_latest_stats(self) -> Dict[str, Any]:
        """Get latest generation statistics."""
        return self.history[-1] if self.history else {}
    
    def export_to_csv(self, filename: str) -> None:
        """Export statistics to CSV file."""
        import pandas as pd
        
        if not self.history:
            return
        
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)


def create_checkpoint(population: np.ndarray, fitness: List[float], 
                     generation: int, config: Dict[str, Any], 
                     checkpoint_path: str) -> None:
    """
    Create a checkpoint of the current state.
    
    Args:
        population: Current population
        fitness: Current fitness values
        generation: Current generation number
        config: Algorithm configuration
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_data = {
        'population': population.tolist(),
        'fitness': fitness,
        'generation': generation,
        'config': config,
        'timestamp': time.time()
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    with open(checkpoint_path, 'w') as f:
        yaml.dump(checkpoint_data, f)


def load_checkpoint(checkpoint_path: str) -> Tuple[np.ndarray, List[float], int, Dict[str, Any]]:
    """
    Load checkpoint data.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (population, fitness, generation, config)
    """
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = yaml.safe_load(f)
    
    population = np.array(checkpoint_data['population'])
    fitness = checkpoint_data['fitness']
    generation = checkpoint_data['generation']
    config = checkpoint_data['config']
    
    return population, fitness, generation, config


def validate_bounds(bounds: List[Tuple[float, float]]) -> bool:
    """
    Validate that bounds are properly formatted.
    
    Args:
        bounds: List of (min, max) tuples
        
    Returns:
        True if bounds are valid
    """
    for bound in bounds:
        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            return False
        if bound[0] >= bound[1]:
            return False
    return True


def clip_to_bounds(individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Clip individual values to specified bounds.
    
    Args:
        individual: Individual to clip
        bounds: List of (min, max) tuples
        
    Returns:
        Clipped individual
    """
    clipped = individual.copy()
    for i, (min_val, max_val) in enumerate(bounds):
        clipped[i] = np.clip(clipped[i], min_val, max_val)
    return clipped