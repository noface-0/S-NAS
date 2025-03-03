#!/usr/bin/env python
"""
Example script for evaluating a discovered neural network architecture.

This script demonstrates how to load a previously discovered architecture from a JSON file
and evaluate it on a specified dataset. It also includes options for exporting the model
to various formats.
"""

import os
import sys
import argparse
import json
import logging
import torch

# Add parent directory to path to access snas package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import S-NAS components
from snas.data.dataset_registry import DatasetRegistry
from snas.architecture.model_builder import ModelBuilder
from snas.search.evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluate.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='S-NAS Architecture Evaluation')
    
    # Required arguments
    parser.add_argument('--architecture', type=str, required=True,
                        help='Path to architecture JSON file')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['cifar10', 'cifar100', 'svhn', 'mnist', 
                                'kmnist', 'qmnist', 'emnist', 'fashion_mnist',
                                'stl10', 'dtd', 'gtsrb'],
                        help='Dataset to evaluate the architecture on')
    
    # Training parameters
    parser.add_argument('--max-epochs', type=int, default=20,
                        help='Maximum epochs for evaluation')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    
    # Export options
    parser.add_argument('--export-model', action='store_true',
                        help='Export the model after evaluation')
    parser.add_argument('--export-format', type=str, default='torchscript',
                        choices=['torchscript', 'onnx', 'quantized', 'mobile', 'all'],
                        help='Format to export the model to')
    parser.add_argument('--export-dir', type=str, default='output/exported_models',
                        help='Directory to save exported models')
    
    # Device selection
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for computation (e.g., "cuda:0", "cpu")')
    
    return parser.parse_args()

def export_model(model, architecture, dataset_config, results, args):
    """Export the model to the specified format."""
    try:
        # Dynamically import the ModelExporter from snas
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from snas.model_exporter import ModelExporter
    except ImportError:
        logger.error("model_exporter module not found. Skipping model export.")
        return None

    # Create model exporter
    os.makedirs(args.export_dir, exist_ok=True)
    exporter = ModelExporter(output_dir=args.export_dir)
    
    # Get input shape from dataset
    input_shape = dataset_config['input_shape']
    
    # Generate model name
    network_type = architecture.get('network_type', 'cnn')
    model_name = f"{network_type}_{args.dataset}_{results['test_acc']:.4f}"
    
    # Export based on selected format
    if args.export_format == 'all':
        logger.info("Exporting model to all formats")
        export_paths = exporter.export_all_formats(
            model, input_shape, architecture, model_name
        )
    else:
        export_paths = {}
        try:
            if args.export_format == 'torchscript':
                path = exporter.export_to_torchscript(model, input_shape, model_name)
            elif args.export_format == 'onnx':
                path = exporter.export_to_onnx(model, input_shape, model_name)
            elif args.export_format == 'quantized':
                path = exporter.export_quantized_model(model, input_shape, model_name)
            elif args.export_format == 'mobile':
                path = exporter.export_model_for_mobile(model, input_shape, model_name)
            
            export_paths[args.export_format] = path
            
            # Generate and save example code
            example_code = exporter.generate_example_code(
                args.export_format, path, input_shape
            )
            example_path = os.path.join(args.export_dir, f"{model_name}_{args.export_format}_example.py")
            with open(example_path, 'w') as f:
                f.write(example_code)
                
            logger.info(f"Model exported to {args.export_format} at: {path}")
            logger.info(f"Example code saved to: {example_path}")
            
        except Exception as e:
            logger.error(f"Error exporting model to {args.export_format}: {e}")
    
    return export_paths

def main():
    """Evaluate an architecture and optionally export it."""
    # Parse arguments
    args = parse_args()
    
    # Load architecture from JSON file
    try:
        with open(args.architecture, 'r') as f:
            architecture = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load architecture from {args.architecture}: {e}")
        return
    
    # Set up dataset registry
    dataset_registry = DatasetRegistry(
        data_dir='./data',
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Get dataset configuration
    dataset_config = dataset_registry.get_dataset_config(args.dataset)
    
    # Ensure architecture has required dataset-specific parameters
    if 'input_shape' not in architecture:
        architecture['input_shape'] = dataset_config['input_shape']
    if 'num_classes' not in architecture:
        architecture['num_classes'] = dataset_config['num_classes']
    
    # Determine device to use
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # Set up model builder
    model_builder = ModelBuilder(device=device)
    
    # Set up evaluator with more epochs for thorough evaluation
    evaluator = Evaluator(
        dataset_registry=dataset_registry,
        model_builder=model_builder,
        device=device,
        max_epochs=args.max_epochs,
        patience=5,  # More patience for final evaluation
        monitor='val_acc'
    )
    
    # Evaluate the architecture
    logger.info(f"Evaluating architecture on {args.dataset} dataset")
    evaluation = evaluator.evaluate(
        architecture, args.dataset, fast_mode=False
    )
    
    # Get the model for potential export
    model, _, _ = model_builder.build_training_components(architecture)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Test accuracy: {evaluation['test_acc']:.4f}")
    print(f"  Best validation accuracy: {evaluation['best_val_acc']:.4f}")
    print(f"  Epochs trained: {evaluation['epochs_trained']}")
    print(f"  Training time: {evaluation['training_time']:.2f} seconds")
    print(f"  Model size: {evaluation['model_size']:.2f} MB")
    
    # Export the model if requested
    if args.export_model:
        logger.info(f"Exporting model to {args.export_format} format")
        export_paths = export_model(model, architecture, dataset_config, evaluation, args)
        
        if export_paths:
            formats = ', '.join(export_paths.keys())
            print(f"  Model exported to: {formats}")
            print(f"  Export directory: {args.export_dir}")

if __name__ == "__main__":
    main()