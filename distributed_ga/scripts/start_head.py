#!/usr/bin/env python3
"""
Start the head node for distributed genetic algorithm.
This script should be run on the Windows laptop (master node).
"""

import os
import sys
import argparse
import logging
import time
import ray
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start Ray head node for distributed GA')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='../config/cluster_config.yaml',
        help='Path to cluster configuration file'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=10001,
        help='Port for Ray head node'
    )
    
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8265,
        help='Port for Ray dashboard'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host IP address'
    )
    
    parser.add_argument(
        '--object-store-memory',
        type=int,
        default=1000000000,
        help='Object store memory in bytes'
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
    
    return parser.parse_args()


def load_cluster_config(config_path):
    """Load cluster configuration."""
    try:
        config = load_config(config_path)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return None


def start_ray_head(config, args):
    """Start Ray head node."""
    try:
        # Get master configuration
        master_config = config.get('master', {})
        ray_config = master_config.get('ray', {})
        
        # Ray initialization parameters
        ray_params = {
            'address': None,  # This makes it a head node
            'port': args.port,
            'dashboard_host': args.host,
            'dashboard_port': args.dashboard_port,
            'object_store_memory': args.object_store_memory,
            'include_dashboard': True,
            'log_to_driver': args.verbose
        }
        
        # Add resource configuration if specified
        resources = master_config.get('resources', {})
        if resources.get('cpu') is not None:
            ray_params['num_cpus'] = resources['cpu']
        if resources.get('gpu') is not None:
            ray_params['num_gpus'] = resources['gpu']
        if resources.get('memory_mb') is not None:
            ray_params['memory'] = resources['memory_mb'] * 1024 * 1024
        
        # Initialize Ray
        logging.info("Starting Ray head node...")
        ray.init(**ray_params)
        
        # Print connection information
        logging.info("Ray head node started successfully!")
        logging.info(f"Dashboard URL: http://{args.host}:{args.dashboard_port}")
        logging.info(f"Head node address: {ray.get_runtime_context().gcs_address}")
        
        # Print worker connection command
        worker_command = f"ray start --address='{ray.get_runtime_context().gcs_address}'"
        logging.info(f"Workers can connect using: {worker_command}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to start Ray head node: {e}")
        return False


def monitor_cluster(config):
    """Monitor cluster status."""
    logging.info("Monitoring cluster status...")
    logging.info("Press Ctrl+C to stop the head node")
    
    try:
        while True:
            # Get cluster status
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Print status
            logging.info("=== Cluster Status ===")
            logging.info(f"Total CPUs: {cluster_resources.get('CPU', 0)}")
            logging.info(f"Available CPUs: {available_resources.get('CPU', 0)}")
            logging.info(f"Total memory: {cluster_resources.get('memory', 0) / (1024**3):.2f} GB")
            logging.info(f"Available memory: {available_resources.get('memory', 0) / (1024**3):.2f} GB")
            logging.info(f"Total nodes: {len(ray.nodes())}")
            
            # Print node information
            for i, node in enumerate(ray.nodes()):
                node_id = node['NodeID']
                alive = node['Alive']
                resources = node['Resources']
                logging.info(f"Node {i+1}: {node_id[:8]}... (Alive: {alive})")
                logging.info(f"  Resources: {resources}")
            
            logging.info("=" * 50)
            
            # Wait before next status update
            time.sleep(30)
            
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"Error monitoring cluster: {e}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("Starting distributed GA head node")
    logger.info(f"Configuration file: {args.config}")
    
    # Load configuration
    config = load_cluster_config(args.config)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        return 1
    
    # Start Ray head node
    if not start_ray_head(config, args):
        logger.error("Failed to start Ray head node. Exiting.")
        return 1
    
    try:
        # Monitor cluster
        monitor_cluster(config)
        
    finally:
        # Shutdown Ray
        logger.info("Shutting down Ray head node...")
        ray.shutdown()
        logger.info("Head node shutdown completed")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())