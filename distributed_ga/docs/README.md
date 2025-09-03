# Distributed Genetic Algorithm Framework

A Ray-based distributed genetic algorithm framework for mathematical modeling optimization problems, designed to run across Windows laptops (head nodes) and Linux servers (worker nodes).

## Features

- **Distributed Computing**: Ray framework for cross-platform distributed computing
- **Genetic Algorithm Components**: 
  - Multiple selection operators (roulette wheel, tournament)
  - Various crossover operators (single-point, uniform, arithmetic)
  - Different mutation operators (Gaussian, uniform)
- **Test Functions**: Classic optimization benchmarks (Rastrigin, Sphere, Rosenbrock, etc.)
- **Performance Monitoring**: Real-time statistics and convergence tracking
- **Fault Tolerance**: Automatic reconnection and checkpointing
- **Docker Support**: Containerized deployment for easy setup

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd distributed_ga

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Head Node (Windows Laptop)

```bash
# Start Ray head node
python scripts/start_head.py --config config/cluster_config.yaml

# The head node will display connection information for workers
```

### 3. Start Worker Nodes (Linux Servers)

```bash
# On each Linux server, start a worker
python scripts/start_worker.py --address ray://HEAD_NODE_IP:10001

# Replace HEAD_NODE_IP with the actual IP address of the head node
```

### 4. Run Optimization Experiment

```bash
# Run an experiment
python scripts/run_experiment.py --config config/ga_config.yaml --function rastrigin --dimensions 10
```

## Configuration

### Genetic Algorithm Configuration (`config/ga_config.yaml`)

```yaml
# Population settings
population:
  size: 100
  elitism_size: 2

# Problem settings
problem:
  dimensions: 10
  bounds: [-5.12, 5.12]
  function_name: "rastrigin"

# Algorithm parameters
algorithm:
  max_generations: 1000
  target_fitness: 1e-6
  convergence_threshold: 1e-8
  convergence_generations: 50

# Genetic operators
operators:
  selection:
    type: "roulette_wheel"  # or "tournament"
  crossover:
    type: "single_point"    # or "uniform", "arithmetic"
    probability: 0.8
  mutation:
    type: "gaussian"        # or "uniform"
    probability: 0.1

# Distributed computing
distributed:
  enabled: true
  num_workers: null  # auto-detect
  batch_size: null   # auto-calculate
```

### Cluster Configuration (`config/cluster_config.yaml`)

```yaml
# Master node (Windows laptop)
master:
  ray:
    port: 10001
    dashboard_port: 8265
  network:
    host: "0.0.0.0"
    allow_external_connections: true

# Worker nodes (Linux servers)
workers:
  - name: "linux_worker_1"
    host: "192.168.1.100"
    username: "user"
    ray:
      num_cpus: null  # auto-detect
      memory: null    # auto-detect
```

## Usage Examples

### Basic Optimization

```python
from distributed_ga import DistributedGeneticAlgorithm
from test_functions import TestFunctions

# Create distributed GA
dga = DistributedGeneticAlgorithm(
    fitness_function=TestFunctions.rastrigin,
    dimensions=10,
    bounds=[(-5.12, 5.12)] * 10,
    population_size=100
)

# Run optimization
result = dga.run(max_generations=1000)

print(f"Best solution: {result['best_individual']}")
print(f"Best fitness: {result['best_fitness']}")

# Cleanup
dga.shutdown()
```

### Custom Operators

```python
from genetic_algorithm import TournamentSelection, UniformCrossover, GaussianMutation

# Create custom operators
selection = TournamentSelection(tournament_size=3)
crossover = UniformCrossover(probability=0.5)
mutation = GaussianMutation(sigma=0.1)

# Use with distributed GA
dga = DistributedGeneticAlgorithm(
    fitness_function=TestFunctions.sphere,
    dimensions=5,
    bounds=[(-10, 10)] * 5,
    selection_operator=selection,
    crossover_operator=crossover,
    mutation_operator=mutation
)
```

### Performance Comparison

```python
# Run distributed experiment
python scripts/run_experiment.py --function rastrigin --dimensions 20 --save-plots

# Run sequential comparison
python scripts/run_experiment.py --function rastrigin --dimensions 20 --no-distributed --save-plots
```

## Docker Deployment

### Build Images

```bash
# Build head node image
docker build -f docker/Dockerfile.head -t distributed_ga:head .

# Build worker node image  
docker build -f docker/Dockerfile.worker -t distributed_ga:worker .
```

### Run with Docker

```bash
# Start head node
docker run -p 10001:10001 -p 8265:8265 distributed_ga:head

# Start worker nodes
docker run -e RAY_HEAD_ADDRESS=ray://HEAD_NODE_IP:10001 distributed_ga:worker
```

## Test Functions

The framework includes several classic optimization benchmarks:

| Function | Global Minimum | Search Domain | Characteristics |
|----------|----------------|---------------|-----------------|
| Sphere | f(0,...,0) = 0 | [-5.12, 5.12]^n | Unimodal, convex |
| Rastrigin | f(0,...,0) = 0 | [-5.12, 5.12]^n | Multimodal |
| Rosenbrock | f(1,...,1) = 0 | [-2.048, 2.048]^n | Non-convex, valley |
| Ackley | f(0,...,0) = 0 | [-32.768, 32.768]^n | Multimodal |
| Griewank | f(0,...,0) = 0 | [-600, 600]^n | Multimodal |
| Schwefel | f(420.97,...) ≈ 0 | [-500, 500]^n | Complex multimodal |

## Performance Monitoring

The framework provides comprehensive performance monitoring:

- **Real-time convergence tracking**
- **Worker load balancing statistics**
- **System resource utilization**
- **Distributed computing metrics**

### Monitoring Dashboard

Access the Ray dashboard at `http://HEAD_NODE_IP:8265` to monitor:
- Cluster status
- Task distribution
- Resource utilization
- Performance metrics

## Fault Tolerance

### Automatic Reconnection

Workers automatically reconnect to the head node if the connection is lost:

```bash
python scripts/start_worker.py --address ray://HEAD_NODE_IP:10001 --auto-reconnect
```

### Checkpointing

Save algorithm state periodically:

```python
result = dga.run(
    max_generations=1000,
    checkpoint_interval=50,
    checkpoint_path="./checkpoints/experiment.yaml"
)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_ga.py -v
pytest tests/test_distributed.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## API Reference

### Core Classes

- `DistributedGeneticAlgorithm`: Main distributed GA implementation
- `GeneticAlgorithm`: Sequential GA implementation
- `FitnessEvaluator`: Remote fitness evaluation actor
- `GeneticOperator`: Remote genetic operations actor

### Operators

- **Selection**: `RouletteWheelSelection`, `TournamentSelection`
- **Crossover**: `SinglePointCrossover`, `UniformCrossover`, `ArithmeticCrossover`
- **Mutation**: `GaussianMutation`, `UniformMutation`

### Test Functions

- `TestFunctions`: Collection of benchmark functions
- `get_test_function(name)`: Factory function for test functions

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check firewall settings and Ray port configuration
2. **Out of Memory**: Reduce population size or batch size
3. **Slow Performance**: Increase number of workers or reduce problem complexity

### Debug Mode

Enable debug logging:

```bash
python scripts/start_head.py --log-level DEBUG
python scripts/start_worker.py --log-level DEBUG
```

### Network Issues

For SSH tunneling through firewalls:

```bash
# On worker machine
ssh -L 10001:HEAD_NODE_IP:10001 user@gateway_server

# Then connect to localhost
python scripts/start_worker.py --address ray://localhost:10001
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{distributed_ga,
  title={Distributed Genetic Algorithm Framework},
  author={Distributed GA Team},
  year={2024},
  url={https://github.com/your-repo/distributed_ga}
}
```