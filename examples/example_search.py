#!/usr/bin/env python
"""
Example script for running a neural architecture search on CIFAR-10.

This script demonstrates how to use S-NAS to search for optimal neural network
architectures on the CIFAR-10 dataset. It includes options for checkpointing
and search resumption.
"""

import os
import sys
import argparse
import logging

# Add parent directory to path to access snas package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import S-NAS components
from snas.data.dataset_registry import DatasetRegistry
from snas.architecture.architecture_space import ArchitectureSpace
from snas.architecture.model_builder import ModelBuilder
from snas.search.evaluator import Evaluator
from snas.search.evolutionary_search import EvolutionarySearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('search.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='S-NAS Example Search')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'mnist', 
                                'kmnist', 'qmnist', 'emnist', 'fashion_mnist',
                                'stl10', 'dtd', 'gtsrb'],
                        help='Dataset to use for architecture search')
    
    # Network type options
    parser.add_argument('--network-type', type=str, default=None,
                        choices=['all', 'cnn', 'mlp', 'enhanced_mlp', 'resnet', 'mobilenet', 
                                'densenet', 'shufflenetv2', 'efficientnet'],
                        help='Type of neural network architecture to search')
    
    # Search parameters
    parser.add_argument('--population-size', type=int, default=20,
                        help='Population size for evolutionary search')
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of generations for evolutionary search')
    parser.add_argument('--checkpoint-frequency', type=int, default=0,
                        help='Save checkpoint every N generations (0 to disable)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint file to resume search from')
    
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
    """Run the search example."""
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
    import torch
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Set up model builder
    model_builder = ModelBuilder(device=device)
    
    # Set up evaluator
    evaluator = Evaluator(
        dataset_registry=dataset_registry,
        model_builder=model_builder,
        device=device,
        max_epochs=args.max_epochs,
        patience=3,  # Early stopping patience
        monitor='val_acc'  # Monitor validation accuracy
    )
    
    # Create evolutionary search
    search = EvolutionarySearch(
        architecture_space=architecture_space,
        evaluator=evaluator,
        dataset_name=args.dataset,
        population_size=args.population_size,
        mutation_rate=0.2,
        crossover_rate=0.5,
        generations=args.generations,
        elite_size=2,
        tournament_size=3,
        metric='val_acc',
        save_history=True,
        checkpoint_frequency=args.checkpoint_frequency,
        output_dir=args.output_dir,
        results_dir=results_dir
    )
    
    # Restrict search to specific network type if provided
    if args.network_type and args.network_type != 'all':
        logger.info(f"Restricting search to network type: {args.network_type}")
        # Override the network_type in the sample_random_architecture method
        original_sample = search.architecture_space.sample_random_architecture
        
        # Create wrapped function that forces network_type
        def sample_with_fixed_type():
            arch = original_sample()
            arch['network_type'] = args.network_type
            return arch
            
        # Replace the method
        search.architecture_space.sample_random_architecture = sample_with_fixed_type
    
    # Run evolutionary search
    fast_mode_generations = 2  # Use fast mode for first 2 generations
    best_architecture, best_fitness, history = search.evolve(
        fast_mode_generations=fast_mode_generations,
        resume_from=args.resume_from
    )
    
    # Print summary of results
    print("\nSearch Results Summary:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Best validation accuracy: {best_fitness:.4f}")
    print(f"  Architecture depth: {best_architecture['num_layers']} layers")
    print(f"  Network type: {best_architecture.get('network_type', 'cnn')}")
    
    # Save best architecture path for reference
    import json
    arch_path = os.path.join(results_dir, f"{args.dataset}_best.json")
    with open(arch_path, 'w') as f:
        json.dump(best_architecture, f, indent=2)
    
    print(f"  Best architecture saved to: {arch_path}")

if __name__ == "__main__":
    main()