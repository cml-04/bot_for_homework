#!/usr/bin/env python3
"""
Start worker nodes for distributed genetic algorithm.
This script can be run on Linux servers (worker nodes).
"""

import os
import sys
import argparse
import logging
import time
import ray
import psutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start Ray worker node for distributed GA')
    
    parser.add_argument(
        '--address',
        type=str,
        required=True,
        help='Ray head node address (e.g., ray://192.168.1.100:10001)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='../config/cluster_config.yaml',
        help='Path to cluster configuration file'
    )
    
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=None,
        help='Number of CPUs to use (default: auto-detect)'
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=0,
        help='Number of GPUs to use'
    )
    
    parser.add_argument(
        '--memory',
        type=int,
        default=None,
        help='Memory limit in bytes (default: auto-detect)'
    )
    
    parser.add_argument(
        '--object-store-memory',
        type=int,
        default=500000000,
        help='Object store memory in bytes'
    )
    
    parser.add_argument(
        '--worker-name',
        type=str,
        default=None,
        help='Worker node name'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--auto-reconnect',
        action='store_true',
        help='Automatically reconnect if connection is lost'
    )
    
    return parser.parse_args()


def get_system_resources():
    """Get available system resources."""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_bytes': psutil.virtual_memory().total,
        'disk_space_bytes': psutil.disk_usage('/').total
    }


def find_worker_config(config, worker_name=None):
    """Find worker configuration by name or hostname."""
    if not config or 'workers' not in config:
        return None
    
    # If worker name is provided, find by name
    if worker_name:
        for worker in config['workers']:
            if worker.get('name') == worker_name:
                return worker
    
    # Try to find by hostname
    import socket
    hostname = socket.gethostname()
    
    for worker in config['workers']:
        if worker.get('host') == hostname:
            return worker
    
    # Return first worker config as fallback
    return config['workers'][0] if config['workers'] else None


def calculate_resource_limits(worker_config, system_resources):
    """Calculate resource limits based on configuration."""
    resources = worker_config.get('resources', {})
    
    # CPU calculation
    max_cpu_percent = resources.get('max_cpu_percent', 90)
    max_cpus = int(system_resources['cpu_count'] * max_cpu_percent / 100)
    
    # Memory calculation
    max_memory_percent = resources.get('max_memory_percent', 80)
    max_memory = int(system_resources['memory_bytes'] * max_memory_percent / 100)
    
    return {
        'num_cpus': max_cpus,
        'memory': max_memory
    }


def start_ray_worker(config, args):
    """Start Ray worker node."""
    try:
        # Find worker configuration
        worker_config = find_worker_config(config, args.worker_name)
        
        # Get system resources
        system_resources = get_system_resources()
        logging.info(f"System resources: {system_resources}")
        
        # Calculate resource limits
        if worker_config:
            resource_limits = calculate_resource_limits(worker_config, system_resources)
            logging.info(f"Resource limits: {resource_limits}")
        else:
            resource_limits = {
                'num_cpus': system_resources['cpu_count'],
                'memory': system_resources['memory_bytes']
            }
            logging.warning("No worker configuration found, using system defaults")
        
        # Ray initialization parameters
        ray_params = {
            'address': args.address,
            'num_cpus': args.num_cpus or resource_limits['num_cpus'],
            'num_gpus': args.num_gpus,
            'memory': args.memory or resource_limits['memory'],
            'object_store_memory': args.object_store_memory,
            'log_to_driver': args.verbose
        }
        
        # Set worker name if provided
        if args.worker_name:
            ray_params['runtime_env'] = {'env_vars': {'RAY_WORKER_NAME': args.worker_name}}
        
        # Initialize Ray worker
        logging.info(f"Connecting to Ray head node at: {args.address}")
        logging.info(f"Worker configuration: {ray_params}")
        
        ray.init(**ray_params)
        
        logging.info("Ray worker connected successfully!")
        logging.info(f"Worker node ID: {ray.get_runtime_context().node_id.hex()}")
        logging.info(f"Worker ID: {ray.get_runtime_context().worker_id.hex()}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to start Ray worker: {e}")
        return False


def monitor_worker(config):
    """Monitor worker status."""
    logging.info("Worker node is running...")
    logging.info("Press Ctrl+C to stop the worker")
    
    try:
        while True:
            # Check if still connected to cluster
            try:
                cluster_resources = ray.cluster_resources()
                logging.info(f"Connected to cluster with {cluster_resources.get('CPU', 0)} total CPUs")
            except Exception as e:
                logging.warning(f"Connection check failed: {e}")
                break
            
            # Wait before next check
            time.sleep(60)
            
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"Error monitoring worker: {e}")


def reconnect_worker(config, args, max_attempts=None):
    """Attempt to reconnect worker with exponential backoff."""
    attempt = 0
    delay = 5  # Initial delay in seconds
    
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        logging.info(f"Reconnection attempt {attempt}")
        
        try:
            # Shutdown existing Ray instance
            if ray.is_initialized():
                ray.shutdown()
            
            # Wait before reconnecting
            time.sleep(delay)
            
            # Try to reconnect
            if start_ray_worker(config, args):
                logging.info("Successfully reconnected to cluster")
                return True
                
        except Exception as e:
            logging.error(f"Reconnection attempt {attempt} failed: {e}")
        
        # Exponential backoff
        delay = min(delay * 2, 300)  # Max 5 minutes
    
    logging.error("Failed to reconnect after maximum attempts")
    return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("Starting distributed GA worker node")
    logger.info(f"Head node address: {args.address}")
    logger.info(f"Configuration file: {args.config}")
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.warning(f"Failed to load configuration: {e}")
        config = None
    
    # Start Ray worker
    if not start_ray_worker(config, args):
        logger.error("Failed to start Ray worker. Exiting.")
        return 1
    
    try:
        while True:
            # Monitor worker
            monitor_worker(config)
            
            # If auto-reconnect is enabled and monitoring exited, try to reconnect
            if args.auto_reconnect:
                logger.info("Attempting to reconnect...")
                if not reconnect_worker(config, args, max_attempts=5):
                    logger.error("Failed to reconnect. Exiting.")
                    break
            else:
                logger.info("Worker disconnected. Exiting.")
                break
                
    finally:
        # Shutdown Ray
        logger.info("Shutting down Ray worker...")
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Worker shutdown completed")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())