"""
Test functions for genetic algorithm optimization.
Common benchmark functions used in evolutionary computation.
"""

import numpy as np
from typing import Union, List


class TestFunctions:
    """Collection of test functions for optimization algorithms."""
    
    @staticmethod
    def rastrigin(x: Union[np.ndarray, List[float]], A: float = 10.0) -> float:
        """
        Rastrigin function - multimodal function with many local minima.
        
        Global minimum: f(0, 0, ..., 0) = 0
        Search domain: [-5.12, 5.12]^n
        
        Args:
            x: Input vector
            A: Parameter (default 10.0)
            
        Returns:
            Function value
        """
        x = np.asarray(x)
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def sphere(x: Union[np.ndarray, List[float]]) -> float:
        """
        Sphere function - simple unimodal function.
        
        Global minimum: f(0, 0, ..., 0) = 0
        Search domain: [-5.12, 5.12]^n
        
        Args:
            x: Input vector
            
        Returns:
            Function value
        """
        x = np.asarray(x)
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x: Union[np.ndarray, List[float]]) -> float:
        """
        Rosenbrock function - classic non-convex optimization problem.
        
        Global minimum: f(1, 1, ..., 1) = 0
        Search domain: [-2.048, 2.048]^n
        
        Args:
            x: Input vector
            
        Returns:
            Function value
        """
        x = np.asarray(x)
        if len(x) < 2:
            raise ValueError("Rosenbrock function requires at least 2 dimensions")
        
        result = 0.0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return result
    
    @staticmethod
    def ackley(x: Union[np.ndarray, List[float]], a: float = 20.0, b: float = 0.2, c: float = 2*np.pi) -> float:
        """
        Ackley function - multimodal function with exponential terms.
        
        Global minimum: f(0, 0, ..., 0) = 0
        Search domain: [-32.768, 32.768]^n
        
        Args:
            x: Input vector
            a, b, c: Function parameters
            
        Returns:
            Function value
        """
        x = np.asarray(x)
        n = len(x)
        
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        
        return term1 + term2 + a + np.exp(1)
    
    @staticmethod
    def griewank(x: Union[np.ndarray, List[float]]) -> float:
        """
        Griewank function - multimodal with product term.
        
        Global minimum: f(0, 0, ..., 0) = 0
        Search domain: [-600, 600]^n
        
        Args:
            x: Input vector
            
        Returns:
            Function value
        """
        x = np.asarray(x)
        
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        
        return sum_term - prod_term + 1
    
    @staticmethod
    def schwefel(x: Union[np.ndarray, List[float]]) -> float:
        """
        Schwefel function - complex multimodal function.
        
        Global minimum: f(420.9687, 420.9687, ..., 420.9687) ≈ 0
        Search domain: [-500, 500]^n
        
        Args:
            x: Input vector
            
        Returns:
            Function value
        """
        x = np.asarray(x)
        n = len(x)
        
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def get_function_info(func_name: str) -> dict:
        """
        Get information about a test function.
        
        Args:
            func_name: Name of the function
            
        Returns:
            Dictionary with function information
        """
        info_dict = {
            'rastrigin': {
                'global_min': 0.0,
                'global_optimum': [0.0],
                'search_domain': [-5.12, 5.12],
                'multimodal': True,
                'description': 'Multimodal function with many local minima'
            },
            'sphere': {
                'global_min': 0.0,
                'global_optimum': [0.0],
                'search_domain': [-5.12, 5.12],
                'multimodal': False,
                'description': 'Simple unimodal convex function'
            },
            'rosenbrock': {
                'global_min': 0.0,
                'global_optimum': [1.0],
                'search_domain': [-2.048, 2.048],
                'multimodal': False,
                'description': 'Non-convex function, valley-shaped'
            },
            'ackley': {
                'global_min': 0.0,
                'global_optimum': [0.0],
                'search_domain': [-32.768, 32.768],
                'multimodal': True,
                'description': 'Multimodal with exponential terms'
            },
            'griewank': {
                'global_min': 0.0,
                'global_optimum': [0.0],
                'search_domain': [-600, 600],
                'multimodal': True,
                'description': 'Multimodal with product term'
            },
            'schwefel': {
                'global_min': 0.0,
                'global_optimum': [420.9687],
                'search_domain': [-500, 500],
                'multimodal': True,
                'description': 'Complex multimodal function'
            }
        }
        
        return info_dict.get(func_name.lower(), {})


# Factory function for easy access
def get_test_function(name: str):
    """
    Get a test function by name.
    
    Args:
        name: Function name
        
    Returns:
        Function object
    """
    functions = {
        'rastrigin': TestFunctions.rastrigin,
        'sphere': TestFunctions.sphere,
        'rosenbrock': TestFunctions.rosenbrock,
        'ackley': TestFunctions.ackley,
        'griewank': TestFunctions.griewank,
        'schwefel': TestFunctions.schwefel
    }
    
    return functions.get(name.lower(), None)