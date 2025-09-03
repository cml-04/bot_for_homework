#!/usr/bin/env python3
"""
Run optimization experiments using the distributed genetic algorithm.
"""

import os
import sys
import argparse
import logging
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, load_config, save_config
from test_functions import get_test_function, TestFunctions
from genetic_algorithm import (
    RouletteWheelSelection, TournamentSelection,
    SinglePointCrossover, UniformCrossover, ArithmeticCrossover,
    GaussianMutation, UniformMutation
)
from distributed_ga import DistributedGeneticAlgorithm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run distributed GA experiments')
    
    parser.add_argument(
        '--config',
        type=str,
        default='distributed_ga/config/ga_config.yaml',
        help='Path to GA configuration file'
    )
    
    parser.add_argument(
        '--function',
        type=str,
        choices=['rastrigin', 'sphere', 'rosenbrock', 'ackley', 'griewank', 'schwefel'],
        help='Test function to optimize (overrides config)'
    )
    
    parser.add_argument(
        '--dimensions',
        type=int,
        help='Problem dimensions (overrides config)'
    )
    
    parser.add_argument(
        '--population-size',
        type=int,
        help='Population size (overrides config)'
    )
    
    parser.add_argument(
        '--max-generations',
        type=int,
        help='Maximum generations (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='distributed_ga/results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated)'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save convergence plots'
    )
    
    parser.add_argument(
        '--compare-sequential',
        action='store_true',
        help='Compare with sequential GA'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--no-distributed',
        action='store_true',
        help='Run without distributed computing (for testing)'
    )
    
    return parser.parse_args()


def create_operators_from_config(config):
    """Create genetic operators from configuration."""
    operators_config = config.get('operators', {})
    
    # Selection operator
    selection_config = operators_config.get('selection', {})
    selection_type = selection_config.get('type', 'roulette_wheel')
    
    if selection_type == 'tournament':
        tournament_size = selection_config.get('tournament_size', 3)
        selection_op = TournamentSelection(tournament_size)
    else:  # roulette_wheel
        selection_op = RouletteWheelSelection()
    
    # Crossover operator
    crossover_config = operators_config.get('crossover', {})
    crossover_type = crossover_config.get('type', 'single_point')
    
    if crossover_type == 'uniform':
        uniform_prob = crossover_config.get('uniform_probability', 0.5)
        crossover_op = UniformCrossover(uniform_prob)
    elif crossover_type == 'arithmetic':
        alpha = crossover_config.get('arithmetic_alpha', 0.5)
        crossover_op = ArithmeticCrossover(alpha)
    else:  # single_point
        crossover_op = SinglePointCrossover()
    
    # Mutation operator
    mutation_config = operators_config.get('mutation', {})
    mutation_type = mutation_config.get('type', 'gaussian')
    
    if mutation_type == 'uniform':
        strength = mutation_config.get('uniform_strength', 0.1)
        mutation_op = UniformMutation(strength)
    else:  # gaussian
        sigma = mutation_config.get('gaussian_sigma', 0.1)
        mutation_op = GaussianMutation(sigma)
    
    return selection_op, crossover_op, mutation_op


def run_experiment(config, args):
    """Run a single experiment."""
    # Override config with command line arguments
    if args.function:
        config['problem']['function_name'] = args.function
    if args.dimensions:
        config['problem']['dimensions'] = args.dimensions
    if args.population_size:
        config['population']['size'] = args.population_size
    if args.max_generations:
        config['algorithm']['max_generations'] = args.max_generations
    
    # Get problem configuration
    problem_config = config['problem']
    function_name = problem_config['function_name']
    dimensions = problem_config['dimensions']
    bounds_range = problem_config['bounds']
    
    # Create bounds list
    if isinstance(bounds_range, list) and len(bounds_range) == 2:
        bounds = [tuple(bounds_range) for _ in range(dimensions)]
    else:
        raise ValueError("Bounds must be a list of [min, max]")
    
    # Get fitness function
    fitness_function = get_test_function(function_name)
    if fitness_function is None:
        raise ValueError(f"Unknown function: {function_name}")
    
    # Get function info
    func_info = TestFunctions.get_function_info(function_name)
    
    logging.info(f"Running experiment: {function_name} ({dimensions}D)")
    logging.info(f"Function info: {func_info}")
    
    # Create operators
    selection_op, crossover_op, mutation_op = create_operators_from_config(config)
    
    # Algorithm parameters
    algorithm_config = config['algorithm']
    population_config = config['population']
    operators_config = config['operators']
    
    # Create distributed GA
    if args.no_distributed:
        from genetic_algorithm import GeneticAlgorithm
        ga = GeneticAlgorithm(
            fitness_function=fitness_function,
            dimensions=dimensions,
            bounds=bounds,
            population_size=population_config['size'],
            selection_operator=selection_op,
            crossover_operator=crossover_op,
            mutation_operator=mutation_op,
            crossover_probability=operators_config['crossover']['probability'],
            mutation_probability=operators_config['mutation']['probability'],
            elitism_size=population_config['elitism_size']
        )
        
        # Run sequential GA
        start_time = time.time()
        result = ga.run(
            max_generations=algorithm_config['max_generations'],
            target_fitness=algorithm_config.get('target_fitness'),
            convergence_threshold=algorithm_config.get('convergence_threshold', 1e-6),
            convergence_generations=algorithm_config.get('convergence_generations', 50)
        )
        end_time = time.time()
        result['total_time'] = end_time - start_time
        result['algorithm_type'] = 'sequential'
        
    else:
        # Distributed GA
        distributed_config = config.get('distributed', {})
        
        dga = DistributedGeneticAlgorithm(
            fitness_function=fitness_function,
            dimensions=dimensions,
            bounds=bounds,
            population_size=population_config['size'],
            selection_operator=selection_op,
            crossover_operator=crossover_op,
            mutation_operator=mutation_op,
            crossover_probability=operators_config['crossover']['probability'],
            mutation_probability=operators_config['mutation']['probability'],
            elitism_size=population_config['elitism_size'],
            num_workers=distributed_config.get('num_workers'),
            batch_size=distributed_config.get('batch_size'),
            island_model=distributed_config.get('island_model', False),
            migration_rate=distributed_config.get('migration_rate', 0.1),
            migration_interval=distributed_config.get('migration_interval', 10)
        )
        
        # Run distributed GA
        result = dga.run(
            max_generations=algorithm_config['max_generations'],
            target_fitness=algorithm_config.get('target_fitness'),
            convergence_threshold=algorithm_config.get('convergence_threshold', 1e-6),
            convergence_generations=algorithm_config.get('convergence_generations', 50)
        )
        result['algorithm_type'] = 'distributed'
        
        # Cleanup
        dga.shutdown()
    
    # Add experiment metadata
    result['experiment_metadata'] = {
        'function_name': function_name,
        'dimensions': dimensions,
        'bounds': bounds,
        'population_size': population_config['size'],
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def save_results(result, output_dir, experiment_name):
    """Save experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_result = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            json_result[key] = value.tolist()
        elif key == 'convergence_data':
            generations, fitness_values = value
            json_result[key] = {
                'generations': generations,
                'fitness_values': fitness_values
            }
        else:
            json_result[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(json_result, f, indent=2, default=str)
    
    logging.info(f"Results saved to: {json_path}")
    
    return json_path


def create_plots(result, output_dir, experiment_name):
    """Create and save plots."""
    try:
        # Convergence plot
        generations, fitness_values = result['convergence_data']
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(generations, fitness_values, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.title(f'Convergence Plot - {experiment_name}')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"{experiment_name}_convergence.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Convergence plot saved to: {plot_path}")
        
        # Performance statistics plot if available
        if 'worker_statistics' in result:
            worker_stats = result['worker_statistics']
            
            # CPU usage plot
            plt.figure(figsize=(12, 8))
            
            # Extract evaluation counts
            eval_counts = [stat['evaluations_count'] for stat in worker_stats['evaluator_stats']]
            worker_ids = [f"Worker {i+1}" for i in range(len(eval_counts))]
            
            plt.subplot(2, 2, 1)
            plt.bar(worker_ids, eval_counts)
            plt.title('Evaluations per Worker')
            plt.xlabel('Worker')
            plt.ylabel('Number of Evaluations')
            plt.xticks(rotation=45)
            
            # Operation counts
            op_counts = [stat['operations_count'] for stat in worker_stats['operator_stats']]
            
            plt.subplot(2, 2, 2)
            plt.bar(worker_ids, op_counts)
            plt.title('Operations per Worker')
            plt.xlabel('Worker')
            plt.ylabel('Number of Operations')
            plt.xticks(rotation=45)
            
            # Total statistics
            plt.subplot(2, 2, 3)
            totals = [
                worker_stats['total_evaluations'],
                worker_stats['total_operations']
            ]
            labels = ['Total Evaluations', 'Total Operations']
            plt.bar(labels, totals)
            plt.title('Total Work Statistics')
            plt.ylabel('Count')
            
            # Runtime information
            plt.subplot(2, 2, 4)
            runtime_info = [
                result['total_time'],
                result['generations_run']
            ]
            runtime_labels = ['Total Time (s)', 'Generations']
            plt.bar(runtime_labels, runtime_info)
            plt.title('Runtime Information')
            
            plt.tight_layout()
            
            stats_plot_path = os.path.join(output_dir, f"{experiment_name}_statistics.png")
            plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Statistics plot saved to: {stats_plot_path}")
        
    except Exception as e:
        logging.warning(f"Failed to create plots: {e}")


def print_summary(result):
    """Print experiment summary."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    metadata = result.get('experiment_metadata', {})
    print(f"Function: {metadata.get('function_name', 'Unknown')}")
    print(f"Dimensions: {metadata.get('dimensions', 'Unknown')}")
    print(f"Population Size: {metadata.get('population_size', 'Unknown')}")
    print(f"Algorithm Type: {result.get('algorithm_type', 'Unknown')}")
    
    print(f"\nBest Fitness: {result['best_fitness']:.6e}")
    print(f"Best Solution: {result['best_individual']}")
    print(f"Generations Run: {result['generations_run']}")
    print(f"Total Time: {result['total_time']:.2f} seconds")
    
    if 'worker_statistics' in result:
        worker_stats = result['worker_statistics']
        print(f"\nDistributed Computing Statistics:")
        print(f"Total Evaluations: {worker_stats['total_evaluations']}")
        print(f"Total Operations: {worker_stats['total_operations']}")
        print(f"Number of Workers: {len(worker_stats['evaluator_stats'])}")
    
    print("="*60)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("Starting distributed GA experiment")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
        
        # Generate experiment name
        if args.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            func_name = args.function or config['problem']['function_name']
            dims = args.dimensions or config['problem']['dimensions']
            args.experiment_name = f"{func_name}_{dims}D_{timestamp}"
        
        logger.info(f"Experiment name: {args.experiment_name}")
        
        # Run experiment
        result = run_experiment(config, args)
        
        # Save results
        save_results(result, args.output_dir, args.experiment_name)
        
        # Create plots
        if args.save_plots:
            create_plots(result, args.output_dir, args.experiment_name)
        
        # Print summary
        print_summary(result)
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())