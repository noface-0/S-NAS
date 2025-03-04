#!/usr/bin/env python
"""
Example script for running a PNAS (Progressive Neural Architecture Search) on CIFAR-10.

This script demonstrates how to use the more faithful implementation of PNAS
which includes a surrogate model for performance prediction as described in 
the original paper by Liu et al. (2018).

Key features:
1. Surrogate model (LSTM) to predict architecture performance without full training
2. Beam search with progressive complexity increase
3. Architecture expansion strategy similar to the original paper
"""

import os
import sys
import argparse
import logging
import torch

# Add parent directory to path to access snas package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import S-NAS components
from snas.data.dataset_registry import DatasetRegistry
from snas.architecture.architecture_space import ArchitectureSpace
from snas.architecture.model_builder import ModelBuilder
from snas.search.evaluator import Evaluator
from snas.search.pnas.pnas_search import PNASSearch
from snas.search.pnas.surrogate_model import SurrogateModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pnas_search.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='PNAS Example Search')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'mnist', 
                                'kmnist', 'qmnist', 'emnist', 'fashion_mnist',
                                'stl10', 'dtd', 'gtsrb'],
                        help='Dataset to use for architecture search')
    
    # PNAS specific parameters
    parser.add_argument('--beam-size', type=int, default=10,
                        help='Number of architectures to keep in the beam')
    parser.add_argument('--max-complexity', type=int, default=3,
                        help='Maximum complexity level to explore (1-3)')
    parser.add_argument('--num-expansions', type=int, default=5,
                        help='Number of expansions per architecture in the beam')
    parser.add_argument('--checkpoint-frequency', type=int, default=1,
                        help='Save checkpoint after each complexity level (0 to disable)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint file to resume search from')
                        
    # PNAS+ENAS combined parameters
    parser.add_argument('--use-shared-weights', action='store_true',
                        help='Enable PNAS+ENAS hybrid mode with shared weights')
    parser.add_argument('--shared-weights-importance', type=float, default=0.5,
                        help='Balance between surrogate model and shared weights (0.0-1.0)')
    
    # Training parameters
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Maximum epochs per architecture evaluation')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save results')
    
    # Device selection
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for computation (e.g., "cuda:0", "cpu")')
    
    return parser.parse_args()

def setup_output_dirs(output_dir):
    """Create output directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

def main():
    """Run the PNAS search example."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    results_dir = setup_output_dirs(args.output_dir)
    
    # Set up dataset registry
    dataset_registry = DatasetRegistry(
        data_dir='./data',
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Get dataset configuration
    dataset_config = dataset_registry.get_dataset_config(args.dataset)
    
    # Set up architecture space
    architecture_space = ArchitectureSpace(
        input_shape=dataset_config['input_shape'],
        num_classes=dataset_config['num_classes']
    )
    
    # Determine device to use
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Set up model builder
    model_builder = ModelBuilder(device=device)
    
    # Set up evaluator with parameter sharing enabled by default
    evaluator = Evaluator(
        dataset_registry=dataset_registry,
        model_builder=model_builder,
        device=device,
        max_epochs=args.max_epochs,
        patience=3,  # Early stopping patience
        monitor='val_acc',  # Monitor validation accuracy
        enable_weight_sharing=True,  # Parameter sharing still enabled for efficiency
        weight_sharing_max_models=100  # Keep up to 100 models in the pool
    )
    
    # Create a new surrogate model (or load existing one if resuming)
    surrogate_model = None
    if args.resume_from and os.path.exists(f"{results_dir}/{args.dataset}_surrogate_model.pt"):
        try:
            surrogate_model = SurrogateModel(device=device)
            surrogate_model.load_model(f"{results_dir}/{args.dataset}_surrogate_model.pt")
            logger.info("Loaded existing surrogate model")
        except Exception as e:
            logger.error(f"Error loading surrogate model: {e}")
            surrogate_model = None
    
    # Create PNAS search
    pnas = PNASSearch(
        architecture_space=architecture_space,
        evaluator=evaluator,
        dataset_name=args.dataset,
        beam_size=args.beam_size,
        num_expansions=args.num_expansions,
        max_complexity=args.max_complexity,
        surrogate_model=surrogate_model,
        predictor_batch_size=100,
        metric='val_acc',
        checkpoint_frequency=args.checkpoint_frequency,
        output_dir=args.output_dir,
        results_dir=results_dir,
        use_shared_weights=args.use_shared_weights,
        shared_weights_importance=args.shared_weights_importance
    )
    
    # Run PNAS search
    best_architecture, best_fitness, history = pnas.search(
        resume_from=args.resume_from
    )
    
    # Print summary of results
    if args.use_shared_weights:
        search_type = "PNAS+ENAS Combined"
    else:
        search_type = "PNAS"
        
    print(f"\n{search_type} Search Results Summary:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Best validation accuracy: {best_fitness:.4f}")
    print(f"  Architecture depth: {best_architecture['num_layers']} layers")
    print(f"  Network type: {best_architecture.get('network_type', 'cnn')}")
    
    if args.use_shared_weights:
        print(f"  Shared weights importance: {args.shared_weights_importance}")
    
    # Save best architecture path for reference
    import json
    arch_path = os.path.join(results_dir, f"{args.dataset}_pnas_best.json")
    with open(arch_path, 'w') as f:
        json.dump(best_architecture, f, indent=2)
    
    print(f"  Best architecture saved to: {arch_path}")

if __name__ == "__main__":
    main()