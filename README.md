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
- **Custom Dataset Support**:
  - CSV-based datasets with image paths and labels
  - Folder-based image datasets with class subfolders
- **Advanced Metrics**: Computes precision, recall, F1, confusion matrices, and ROC curves
- **Model Export**: Exports models to ONNX, TorchScript, quantized, and mobile formats
- **Distributed Evaluation**: Distributes model training across multiple GPUs
- **Visualization**: Provides rich visualizations of search progress and architecture performance
- **Streamlit Interface**: User-friendly web interface for controlling the search process

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
  - `utils/`: Job distributor for parallel evaluation
  - `visualization/`: Visualization utilities
- `app.py`: Streamlit application
- `main.py`: Command-line interface
- `examples/`: Example scripts

## Architecture Space

S-NAS explores architectures with the following parameters and neural network types:

### Network Types

- **CNN**: Standard convolutional neural networks
- **MLP**: Multi-layer perceptrons (fully-connected networks)
- **ResNet**: Residual networks with skip connections
- **MobileNet**: Networks with depthwise separable convolutions

### Parameters

- Number of layers: 2-8
- Filters per layer (CNN/ResNet/MobileNet): 16, 32, 64, 128, 256
- Hidden units per layer (MLP): 64, 128, 256, 512, 1024
- Kernel sizes: 3, 5, 7
- Width multiplier (MobileNet): 0.5, 0.75, 1.0, 1.25
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
- **Early Stopping**: Configure patience and monitoring metric to optimize training time
- **Data Loading**: Adjust num_workers based on your CPU capabilities for faster data loading
- **Batch Size**: Larger batch sizes can speed up training on powerful GPUs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on concepts from ENAS, DARTS, and other neural architecture search papers
- Inspired by evolutionary approaches to hyperparameter optimization