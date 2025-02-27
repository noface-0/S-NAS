# S-NAS: Simple Neural Architecture Search

S-NAS is a streamlined system that automates the discovery of optimal neural network architectures for specific datasets. Rather than manually designing neural networks, S-NAS efficiently explores different architecture configurations to find ones that perform best on predefined benchmark datasets.

## Features

- **Evolutionary Search**: Uses genetic algorithms to efficiently explore the architecture space
- **Multiple Datasets**: Works with standard datasets (CIFAR-10, MNIST, Fashion-MNIST)
- **Distributed Evaluation**: Distributes model training across multiple GPUs
- **Visualization**: Provides rich visualizations of search progress and architecture performance
- **Streamlit Interface**: User-friendly web interface for controlling the search process
- **Model Export**: Exports discovered architectures as reusable PyTorch models

## How It Works

S-NAS uses an evolutionary approach to neural architecture search:

1. **Initialization**: Creates a population of random architecture configurations
2. **Evaluation**: Trains and evaluates each architecture on the target dataset
3. **Selection**: Selects the better-performing architectures as parents
4. **Evolution**: Creates a new generation through crossover and mutation
5. **Iteration**: Repeats the process for multiple generations
6. **Export**: Provides the best-discovered architecture for practical use

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/snas.git
   cd snas
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Streamlit Interface

The easiest way to use S-NAS is through the Streamlit interface:

```bash
streamlit run app.py
```

This will open a web interface where you can:
- Select a dataset
- Configure search parameters
- Run the search process
- Visualize results
- Export discovered architectures

### Command Line

You can also run S-NAS from the command line for batch processing:

```bash
python main.py --dataset cifar10 --population-size 20 --generations 10 --gpu-ids 0,1
```

Common options:
- `--dataset`: Dataset to use (`cifar10`, `mnist`, `fashion_mnist`)
- `--population-size`: Population size for evolutionary search
- `--generations`: Number of generations to evolve
- `--gpu-ids`: Comma-separated list of GPU IDs to use
- `--output-dir`: Directory to save results
- `--evaluate`: Path to an architecture JSON file for evaluation only

### Programmatic Usage

You can also use S-NAS components programmatically in your own Python code:

```python
from snas.data.dataset_registry import DatasetRegistry
from snas.architecture.architecture_space import ArchitectureSpace
from snas.search.evolutionary_search import EvolutionarySearch
# ...

# Initialize components
dataset_registry = DatasetRegistry()
architecture_space = ArchitectureSpace(input_shape=(3, 32, 32), num_classes=10)
# ...

# Run evolutionary search
search = EvolutionarySearch(architecture_space, evaluator, "cifar10")
best_architecture, best_fitness, history = search.evolve()
```

Check the `examples` directory for complete usage examples.

## Project Structure

- `snas/`: Main package
  - `architecture/`: Architecture space and model builder
  - `data/`: Dataset registry
  - `search/`: Evolutionary search and evaluator
  - `utils/`: Job distributor for parallel evaluation
  - `visualization/`: Visualization utilities
- `app.py`: Streamlit application
- `main.py`: Command-line interface
- `examples/`: Example scripts

## Architecture Space

S-NAS explores architectures with the following parameters:

- Number of layers: 2-8
- Filters per layer: 16, 32, 64, 128, 256
- Kernel sizes: 3, 5, 7
- Activations: ReLU, Leaky ReLU, ELU, SELU
- Batch normalization: Yes/No
- Dropout rate: 0.0, 0.1, 0.2, 0.3, 0.5
- Skip connections: Yes/No
- Learning rate: 0.1, 0.01, 0.001, 0.0001
- Optimizer: SGD, Adam, AdamW

This space can be customized in `architecture_space.py`.

## Examples

### Running a Search

```python
# Example script to run a search on CIFAR-10
python examples/example_search.py
```

### Evaluating an Architecture

```python
# Evaluate a discovered architecture on MNIST
python examples/example_evaluate.py --architecture output/best_architecture.json --dataset mnist
```

## Results Visualization

S-NAS provides several visualization options:

- **Search Progress**: Shows best and average fitness over generations
- **Architecture Visualization**: Network visualization of the discovered architecture
- **Training Curves**: Training and validation loss/accuracy curves
- **Parameter Importance**: Analysis of which parameters contribute most to performance

## Exporting Models

After discovering a good architecture, you can export it as a standalone PyTorch model:

```python
python examples/example_evaluate.py --architecture output/best_architecture.json --dataset cifar10 --export-model
```

This will generate a Python file containing a self-contained PyTorch model that you can use in your own projects.

## Performance Tips

- **Fast Mode**: Early generations can use a reduced training protocol for faster exploration
- **GPU Parallelization**: Use multiple GPUs to evaluate architectures in parallel
- **Population Size**: Larger populations explore more architectures but take longer
- **Generations**: More generations allow for finer optimization but take longer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on concepts from ENAS, DARTS, and other neural architecture search papers
- Inspired by evolutionary approaches to hyperparameter optimization

## References

1. Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., & Dean, J. (2018). *Efficient Neural Architecture Search via Parameter Sharing*. Proceedings of the 35th International Conference on Machine Learning, PMLR 80:4095-4104.

2. Zoph, B., & Le, Q. V. (2017). *Neural Architecture Search with Reinforcement Learning*. International Conference on Learning Representations (ICLR). Retrieved from https://arxiv.org/abs/1611.01578

3. Liu, C., Zoph, B., Neumann, M., Shlens, J., Hua, W., Li, L. J., Fei-Fei, L., Yuille, A., Huang, J., & Murphy, K. (2018). *Progressive Neural Architecture Search*. European Conference on Computer Vision (ECCV). Retrieved from https://arxiv.org/abs/1802.03268

4. Zoph, B., & Le, Q. V. (2017). *Learning Transferable Architectures for Scalable Image Recognition*. Berkeley Deep RL Course. Retrieved from https://rll.berkeley.edu/deeprlcoursesp17/docs/quoc_barret.pdf
