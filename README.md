# S-NAS: Simple Neural Architecture Search

S-NAS is a streamlined system that automates the discovery of optimal neural network architectures for specific datasets. Rather than manually designing neural networks, S-NAS efficiently explores different architecture configurations to find ones that perform best on predefined benchmark datasets.

## Features

- **Evolutionary Search**: Uses genetic algorithms to efficiently explore the architecture space
- **Multiple Neural Network Types**: Supports CNNs, MLPs, ResNets, and MobileNets
- **Multiple Datasets**: Works with standard datasets:
  - CIFAR-10 & CIFAR-100 (32×32 RGB images)
  - SVHN (Street View House Numbers, 32×32 RGB images)
  - MNIST (handwritten digits, 28×28 grayscale images)
  - KMNIST (Japanese characters, 28×28 grayscale images)
  - QMNIST (extended MNIST, 28×28 grayscale images)
  - EMNIST (extended MNIST with letters, 28×28 grayscale images)
  - Fashion-MNIST (fashion items, 28×28 grayscale images)
  - STL-10 (higher resolution object images, 96×96 RGB images)
  - DTD (Describable Textures Dataset, 47 texture categories)
  - GTSRB (German Traffic Sign Recognition Benchmark, 43 traffic sign classes)
- **Custom Dataset Support**:
  - CSV-based datasets with image paths and labels
  - Folder-based image datasets with class subfolders
- **Advanced Metrics**: Computes precision, recall, F1, confusion matrices, and ROC curves
- **Model Export**: Exports models to ONNX, TorchScript, quantized, and mobile formats
- **Checkpoint System**: Automatic state saving and recovery for long-running searches
- **Distributed Evaluation**: Distributes model training across multiple GPUs
- **Visualization**: Provides rich visualizations of search progress and architecture performance
- **Streamlit Interface**: User-friendly web interface for controlling the search process

## How It Works

S-NAS combines evolutionary algorithms with advanced optimization techniques for efficient neural architecture search:

### Core Search Algorithm

The search process follows these steps:

1. **Initialization**:
   - Creates a population of random architecture configurations
   - Each architecture is defined by parameters like network type, layers, filters, activations, etc.
   - Initial architectures follow a progressive complexity approach (starting simpler)

2. **Evaluation**:
   - Trains and evaluates each architecture on the target dataset
   - Uses parameter sharing to avoid training from scratch when possible
   - Records performance metrics (validation accuracy, loss) as fitness scores
   - Caches results to avoid re-evaluating identical architectures

3. **Selection**:
   - Uses tournament selection to choose parent architectures
   - Better-performing architectures have higher chances of being selected
   - Elite preservation keeps the best architectures unchanged across generations

4. **Evolution**:
   - Creates new architectures through crossover (combining parts of parent architectures)
   - Applies controlled random mutations to introduce diversity
   - Mutation rates are adjustable and target different aspects of the architecture
   - Progressive complexity constraints ensure appropriate architecture scale

5. **Iteration**:
   - Repeats the process for multiple generations
   - Complexity gradually increases as search progresses
   - Weight sharing becomes more effective as the pool of trained models grows
   - Population diversity is monitored to prevent premature convergence

6. **Export**:
   - Returns the best-discovered architecture after all generations
   - Can export to multiple model formats (ONNX, TorchScript, etc.)

### Parameter Sharing (ENAS)

Parameter sharing significantly speeds up evaluation:

- Maintains pools of trained weights for each network type
- When evaluating a new architecture:
  1. Calculates similarity scores with previously trained architectures
  2. Selects the most compatible weights based on architectural similarity and performance
  3. Transfers weights where layer shapes match
  4. Initializes random weights only for incompatible layers
  5. Rebuilds optimizer state for the combined model
- Prioritizes high-performing models in the sharing pool
- Separate pools are maintained for different network types (CNN, MLP, etc.)

### Progressive Search (PNAS)

Progressive complexity growth makes exploration more efficient:

- Defines three complexity levels for architectures:
  - **Level 1**: 2-3 layers, basic network types (CNN, MLP), simple components
  - **Level 2**: 3-5 layers, more network types (ResNet, Enhanced MLP), moderate complexity
  - **Level 3**: Full architecture space with all network types and components

- Search proceeds through these phases:
  1. Early generations focus on finding good basic architectures
  2. Middle generations refine and expand promising structures
  3. Later generations explore the full architecture space
  
- For each complexity level, constraints are placed on:
  - Number of layers
  - Network types allowed
  - Layer configurations (filters, kernel sizes, etc.)
  - Advanced features (skip connections, normalization, etc.)

 **Note on Progressive Search and MLP Bias**

  When using progressive search with datasets like MNIST, you may notice a bias toward MLP architectures in the search results. This occurs because:

  1. Progressive search starts with simpler architectures and gradually increases complexity
  2. For grayscale, low-resolution datasets like MNIST, MLPs can achieve good accuracy early in the search
  3. Once MLPs dominate the early population, evolutionary pressure tends to maintain this architecture type

  If you want to explore a wider variety of architectures (especially CNNs) on simpler datasets, consider:
  - Disabling progressive search with --enable-progressive=False
  - Explicitly selecting a network type with --network-type cnn
  - Running more generations to allow the search to explore more complex architectures

  For complex image datasets like CIFAR-10, this bias is less pronounced as CNNs have a natural advantage over MLPs.

  ---

### Exploration Strategies

To ensure thorough search space coverage:

- **Mutation Mechanism**: The mutation system applies stochastic changes to architecture parameters:
  - Each parameter has an independent chance (controlled by mutation_rate) of being modified
  - Network type has a lower mutation probability (0.2 × mutation_rate) to prevent disruptive changes
  - Layer-specific parameters can be added/removed when layer count changes
  - Numeric parameters are perturbed within their allowed ranges
  - Categorical parameters (activations, etc.) randomly select from available options
  - All mutations respect architectural constraints to ensure validity
- **Diversity Tracking**: The system quantifies population diversity at each generation:
  - Calculates Shannon entropy across all architecture parameters
  - Monitors distribution of parameter values in the population
  - Tracks diversity metrics over time to detect convergence
  - Logs diversity scores that can be visualized to understand search dynamics
  - Higher scores indicate more diverse exploration of the search space
- **Tournament Selection**: Uses competitive selection to balance exploration and exploitation:
  - Randomly samples a small subset (tournament_size) of architectures
  - Selects the best performer from each tournament as a parent
  - Smaller tournament sizes maintain diversity but slow convergence
  - Larger tournament sizes increase selection pressure toward top performers
  - Creates a probabilistic process where better architectures are more likely to reproduce

- **Fast Mode**: Accelerates early exploration phases:
  - Reduces the number of training epochs for evaluations
  - Limits the number of batches processed during early generations
  - Enables broader exploration of the architecture space
  - Full evaluation is applied to the final best architecture

- **Crossover Strategy**: Intelligently combines parent architectures to create offspring:
  - For compatible parents (same network type), performs parameter-wise crossover
  - For incompatible parents (different network types), applies mutation to one parent
  - Uses single-point crossover for layer-specific parameters (filters, activations, etc.)
  - Randomly selects parameters from either parent for global settings
  - Ensures architectural consistency after recombination
  - Probabilistically chooses between crossover and mutation-only reproduction

### Performance Optimizations

Additional optimizations include:

- **Caching**: Identical architectures are evaluated only once
- **Early Stopping**: Training uses patience-based stopping to avoid wasting computation
- **GPU Parallelization**: Multiple architectures can be evaluated in parallel across GPUs
- **Checkpoint System**: Search state can be saved and resumed at any point

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:

``` bash
   git clone https://github.com/noface-0/S-NAS.git
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
- `--network-type`: Network architecture type (`all`, `cnn`, `mlp`, `resnet`, `mobilenet`)
- `--population-size`: Population size for evolutionary search
- `--generations`: Number of generations to evolve
- `--gpu-ids`: Comma-separated list of GPU IDs to use
- `--output-dir`: Directory to save results
- `--evaluate`: Path to an architecture JSON file for evaluation only
- `--patience`: Early stopping patience (number of epochs without improvement)
- `--min-delta`: Minimum change to qualify as improvement for early stopping
- `--monitor`: Metric to monitor for early stopping ('val_acc' or 'val_loss')
- `--num-workers`: Number of worker threads for data loading
- `--checkpoint-frequency`: Save a checkpoint every N generations (0 to disable)
- `--resume-from`: Path to a checkpoint file to resume search from
- `--weight-sharing-max-models`: Maximum number of models in the weight sharing pool (default: 100)

## Custom Datasets

S-NAS supports using your own datasets in two formats:

### CSV-based Datasets

You can use a CSV file with columns for image paths and labels:

```bash
python main.py --custom-csv-dataset data/my_dataset.csv --custom-dataset-name my_dataset --image-size 64x64
```

The CSV file should have at least two columns:

- An image column (default: 'image') with relative or absolute image paths
- A label column (default: 'label') with class labels

### Folder-based Datasets

You can also use a folder structure with class subfolders:

```bash
python main.py --custom-folder-dataset data/images --custom-dataset-name my_images --image-size 224x224
```

The folder should have this structure:

```
data/images/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Model Export

S-NAS can export discovered architectures in various formats for deployment:

### TorchScript Export

```bash
python main.py --dataset cifar10 --export-model --export-format torchscript
```

### ONNX Export

```bash
python main.py --dataset mnist --export-model --export-format onnx
```

### Quantized Model Export

```bash
python main.py --dataset fashion_mnist --export-model --export-format quantized
```

### Mobile-Optimized Export

```bash
python main.py --dataset cifar100 --export-model --export-format mobile
```

### All Formats

```bash
python main.py --dataset cifar10 --export-model --export-format all
```

Each exported model comes with an example Python script showing how to use it.

## Project Structure

- `snas/`: Main package
  - `architecture/`: Architecture space and model builder
  - `data/`: Dataset registry
  - `search/`: Evolutionary search and evaluator
  - `utils/`: Utility modules
    - `job_distributor.py`: Parallel evaluation across multiple GPUs
    - `exceptions.py`: Custom exception classes for structured error handling
    - `state_manager.py`: Checkpoint system for saving and resuming searches
  - `visualization/`: Visualization utilities
- `app.py`: Streamlit application
- `main.py`: Command-line interface
- `examples/`: Example scripts

## Architecture Space

S-NAS explores architectures with the following parameters and neural network types:

### Network Types

- **CNN**: Standard convolutional neural networks
- **MLP**: Multi-layer perceptrons (fully-connected networks)
- **Enhanced MLP**: MLP with residual connections and layer normalization
- **ResNet**: Residual networks with skip connections
- **MobileNet**: Networks with depthwise separable convolutions
- **DenseNet**: Networks with dense connectivity pattern
- **ShuffleNetV2**: Networks with channel split and shuffle operations
- **EfficientNet**: Scalable networks with mobile inverted bottleneck convolutions

### Parameters

#### Shared Parameters
- Number of layers: 2-50 (for ResNet, each "layer" is a residual block containing 2-3 actual layers)
- Batch normalization: Yes/No
- Dropout rate: 0.0, 0.1, 0.2, 0.3, 0.5
- Learning rate: 0.1, 0.01, 0.001, 0.0001
- Optimizer: SGD, Adam, AdamW

#### CNN/ResNet/MobileNet Parameters
- Filters per layer: 16, 32, 64, 128, 256
- Kernel sizes: 3, 5, 7
- Activations: ReLU, Leaky ReLU, ELU, SELU
- Skip connections: Yes/No

#### MLP Parameters
- Hidden units per layer: 64, 128, 256, 512, 1024
- Activations: ReLU, Leaky ReLU, ELU, SELU

#### Enhanced MLP Parameters
- Hidden units per layer: 64, 128, 256, 512, 1024
- Activations: ReLU, Leaky ReLU, ELU, SELU, GELU
- Layer normalization: Yes/No
- Residual connections: Yes/No

#### MobileNet Parameters
- Width multiplier: 0.5, 0.75, 1.0, 1.25

#### DenseNet Parameters
- Growth rate: 12, 24, 32, 48
- Block configuration: Various arrangements of layers per block
- Compression factor: 0.5, 0.7, 0.8
- Bottleneck size: 2, 4

#### ShuffleNetV2 Parameters
- Width multiplier: 0.5, 0.75, 1.0, 1.25, 1.5
- Channel configurations: Different for each scale
- Blocks per stage: Various configurations

#### EfficientNet Parameters
- Width factor: 0.5, 0.75, 1.0, 1.25, 1.5
- Depth factor: 0.8, 1.0, 1.2, 1.4
- Squeeze-and-excitation ratio: 0.0, 0.125, 0.25

This space can be customized in `architecture_space.py`.

## Examples

### Running a Search

```python
# Example script to run a search on CIFAR-10
python examples/example_search.py
```

### With Checkpointing

```python
# Run search with checkpoints every 2 generations
python examples/example_search.py --checkpoint-frequency 2
```

### Resuming a Search

```python
# Resume from a previous checkpoint
python examples/example_search.py --resume-from output/results/cifar10_checkpoint_gen5_20250226-123456.pkl
```

### Running a Search

```python
# Parameter sharing and progressive search are enabled by default
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

## Performance Features

S-NAS incorporates several performance optimizations by default:

- **Parameter Sharing (ENAS)**: Weight sharing between models for much faster search (Pham et al., 2018)
- **Progressive Search**: Gradual complexity increase for more efficient exploration (Liu et al., 2018)
- **Fast Mode**: Early generations use a reduced training protocol for faster exploration
- **Gradient Clipping**: Automatic gradient clipping prevents numerical instability issues

Additional performance tips:

- **GPU Parallelization**: Use multiple GPUs to evaluate architectures in parallel
- **Population Size**: Larger populations explore more architectures but take longer
- **Generations**: More generations allow for finer optimization but take longer
- **Early Stopping**: Configure patience and monitoring metric to optimize training time
- **Data Loading**: Adjust num_workers based on your CPU capabilities for faster data loading
- **Batch Size**: Larger batch sizes can speed up training on powerful GPUs
- **Checkpointing**: Use `--checkpoint-frequency` to save progress periodically during long runs
- **Resume Capability**: If a run is interrupted, use `--resume-from` to continue from a checkpoint

### Advanced Search Techniques

S-NAS incorporates two key efficiency technologies from recent research papers by default:

#### Parameter Sharing (ENAS)

Parameter sharing based on the ENAS paper by Pham et al. (2018) significantly speeds up architecture search by reusing weights between similar architectures:

1. When a new model is created, it automatically reuses weights from previously trained models of the same type
2. Weights are transferred where layer shapes match, with new random weights used for incompatible layers
3. The best-performing model weights are prioritized for sharing
4. The system maintains separate weight pools for different network types (CNN, MLP, etc.)

Parameter sharing typically provides a 2-5x speedup for the search process, while maintaining comparable accuracy in discovering effective architectures.

#### Progressive Search

Progressive neural architecture search based on the paper by Liu et al. (2018) makes the search more efficient by gradually increasing architecture complexity:

1. The search begins with simple architectures (fewer layers, basic components)
2. As search progresses, complexity gradually increases in predefined levels
3. Early generations focus on finding good basic structures with fewer parameters
4. Later generations refine those structures with more advanced components

This leads to:
- More efficient exploration of the architecture space
- Better architectures found with the same computational budget
- Reduced chance of getting stuck in local optima

S-NAS combines both these techniques by default, providing state-of-the-art efficiency for neural architecture search.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## TODO List

1. **Add zero-shot entropy metric**
   - Implement entropy calculation based on Zen-NAS paper
   - Create pre-screening function to evaluate architectures without training
   - Set appropriate thresholds for filtering architectures

2. **Optimize evaluation pipeline**
   - Add pre-screening step before full training evaluation
   - Implement early termination for poor-performing models
   - Create fast evaluation mode with reduced epochs

3. **Improve error reporting**
   - Add detailed error messages in evaluation functions
   - Implement structured logging for architecture failures
   - Create error visualization in web interface

4. **Implement efficient architecture comparison**
   - Replace JSON serialization with hash-based tracking
   - Add fast structure similarity calculation
   - Create architecture fingerprinting function

5. **Add support for Vision Transformers**
   - Implement ViT model in model_builder.py
   - Add transformer-specific parameters to architecture space
   - Create appropriate mutation operators for attention mechanisms

6. **Add support for MLP-Mixer architectures**
   - Implement MLP-Mixer model structure
   - Add token/channel mixing parameters to architecture space
   - Create specialized evaluation metrics for MLP-Mixer performance

7. **Add Custom Dataset Support to GUI**
   - Add file upload components to the sidebar
   - Create a new section for dataset configuration options
   - Implement validation for uploaded files
  
8. **Add Checkpoint Management to GUI**
   - Add checkpoint frequency slider to sidebar
   - Create a dropdown for available checkpoints
   - Add resume button to pick up from existing checkpoint

9. **Expand GPU Management in GUI**
   - Expand GPU selection UI to include more distribution options
   - Implement progress tracking for distributed jobs



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268)

2. [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)

3. [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559)

4. [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)
