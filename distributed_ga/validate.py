#!/usr/bin/env python3
"""
Validation script for the distributed genetic algorithm framework.
This script validates that all components work correctly.
"""

import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def validate_imports():
    """Validate that all modules can be imported."""
    print("Validating imports...")
    
    try:
        from genetic_algorithm import GeneticAlgorithm
        from test_functions import TestFunctions, get_test_function
        from utils import setup_logging, load_config
        from distributed_ga import DistributedGeneticAlgorithm
        print("  ✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def validate_configuration():
    """Validate configuration files."""
    print("Validating configuration files...")
    
    try:
        # Load GA config
        with open('distributed_ga/config/ga_config.yaml', 'r') as f:
            ga_config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['population', 'problem', 'algorithm', 'operators', 'distributed']
        for section in required_sections:
            if section not in ga_config:
                print(f"  ✗ Missing section in GA config: {section}")
                return False
        
        # Load cluster config
        with open('distributed_ga/config/cluster_config.yaml', 'r') as f:
            cluster_config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['master', 'workers']
        for section in required_sections:
            if section not in cluster_config:
                print(f"  ✗ Missing section in cluster config: {section}")
                return False
        
        print("  ✓ Configuration files valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration validation failed: {e}")
        return False

def validate_test_functions():
    """Validate test functions."""
    print("Validating test functions...")
    
    try:
        from test_functions import TestFunctions, get_test_function
        
        functions = ['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank', 'schwefel']
        
        for func_name in functions:
            func = get_test_function(func_name)
            if func is None:
                print(f"  ✗ Function not found: {func_name}")
                return False
            
            # Test function call
            result = func([0.0, 0.0])
            if not isinstance(result, (int, float)):
                print(f"  ✗ Function {func_name} returned invalid type: {type(result)}")
                return False
            
            # Test function info
            info = TestFunctions.get_function_info(func_name)
            if not info or 'global_min' not in info:
                print(f"  ✗ Function info missing for: {func_name}")
                return False
        
        print(f"  ✓ All {len(functions)} test functions validated")
        return True
        
    except Exception as e:
        print(f"  ✗ Test function validation failed: {e}")
        return False

def validate_genetic_algorithm():
    """Validate genetic algorithm components."""
    print("Validating genetic algorithm...")
    
    try:
        from genetic_algorithm import GeneticAlgorithm
        from test_functions import TestFunctions
        
        # Create GA instance
        ga = GeneticAlgorithm(
            fitness_function=TestFunctions.sphere,
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            population_size=10
        )
        
        # Test population initialization
        if ga.population.shape != (10, 2):
            print("  ✗ Population initialization failed")
            return False
        
        # Test single generation
        ga.evolve_generation()
        if ga.generation != 1:
            print("  ✗ Generation evolution failed")
            return False
        
        # Test short run
        result = ga.run(max_generations=5)
        required_keys = ['best_individual', 'best_fitness', 'generations_run']
        for key in required_keys:
            if key not in result:
                print(f"  ✗ Missing result key: {key}")
                return False
        
        print("  ✓ Genetic algorithm components validated")
        return True
        
    except Exception as e:
        print(f"  ✗ Genetic algorithm validation failed: {e}")
        return False

def validate_file_structure():
    """Validate file structure."""
    print("Validating file structure...")
    
    required_files = [
        'distributed_ga/src/__init__.py',
        'distributed_ga/src/genetic_algorithm.py',
        'distributed_ga/src/distributed_ga.py',
        'distributed_ga/src/test_functions.py',
        'distributed_ga/src/utils.py',
        'distributed_ga/config/ga_config.yaml',
        'distributed_ga/config/cluster_config.yaml',
        'distributed_ga/scripts/start_head.py',
        'distributed_ga/scripts/start_worker.py',
        'distributed_ga/scripts/run_experiment.py',
        'distributed_ga/tests/test_ga.py',
        'distributed_ga/tests/test_distributed.py',
        'distributed_ga/docker/Dockerfile.head',
        'distributed_ga/docker/Dockerfile.worker',
        'distributed_ga/docs/README.md',
        'distributed_ga/docs/setup_guide.md',
        'distributed_ga/docs/performance_analysis.md',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("  ✗ Missing files:")
        for file_path in missing_files:
            print(f"    - {file_path}")
        return False
    
    print(f"  ✓ All {len(required_files)} required files present")
    return True

def validate_docker_files():
    """Validate Docker configuration."""
    print("Validating Docker files...")
    
    try:
        # Check Dockerfile.head
        with open('distributed_ga/docker/Dockerfile.head', 'r') as f:
            head_content = f.read()
            if 'FROM python:' not in head_content:
                print("  ✗ Dockerfile.head missing Python base image")
                return False
            if 'EXPOSE 10001 8265' not in head_content:
                print("  ✗ Dockerfile.head missing port exposure")
                return False
        
        # Check Dockerfile.worker
        with open('distributed_ga/docker/Dockerfile.worker', 'r') as f:
            worker_content = f.read()
            if 'FROM python:' not in worker_content:
                print("  ✗ Dockerfile.worker missing Python base image")
                return False
        
        print("  ✓ Docker files validated")
        return True
        
    except Exception as e:
        print(f"  ✗ Docker validation failed: {e}")
        return False

def main():
    """Main validation function."""
    print("=" * 60)
    print("DISTRIBUTED GENETIC ALGORITHM FRAMEWORK VALIDATION")
    print("=" * 60)
    
    validations = [
        validate_file_structure,
        validate_imports,
        validate_configuration,
        validate_test_functions,
        validate_genetic_algorithm,
        validate_docker_files
    ]
    
    passed = 0
    total = len(validations)
    
    for validation in validations:
        if validation():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"VALIDATION RESULTS: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("The distributed genetic algorithm framework is ready for deployment.")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please fix the issues before deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(main())