#!/usr/bin/env python
"""
Main entry point for S-NAS.

This script provides a command-line interface for running S-NAS experiments.
"""

import os
import sys
import json
import argparse
import logging
import torch
import time
import pickle
from typing import Dict, Any

# Import S-NAS components
from snas.data.dataset_registry import DatasetRegistry
from snas.architecture.architecture_space import ArchitectureSpace
from snas.architecture.model_builder import ModelBuilder
from snas.search.evaluator import Evaluator
from snas.search.evolutionary_search import EvolutionarySearch
from snas.utils.job_distributor import JobDistributor, ParallelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('snas.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='S-NAS: Simple Neural Architecture Search')
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist', 'fashion_mnist'],
                        help='Dataset to use for architecture search')
    
    # Search parameters
    parser.add_argument('--population-size', type=int, default=20,
                        help='Population size for evolutionary search')
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of generations for evolutionary search')
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                        help='Mutation rate for evolutionary search')
    parser.add_argument('--elite-size', type=int, default=2,
                        help='Number of top architectures to preserve unchanged')
    
    # Training parameters
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Maximum epochs per architecture evaluation')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--fast-mode-gens', type=int, default=2,
                        help='Number of generations to use fast evaluation mode')
    
    # Hardware options
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda:0", "cpu")')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker threads for parallel evaluation')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for the experiment (default: auto-generated)')
    
    # Evaluation mode
    parser.add_argument('--evaluate', type=str, default=None,
                        help='Path to architecture JSON file for evaluation only (no search)')
    
    return parser.parse_args()

def setup_output_dirs(output_dir):
    """Create output directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    return results_dir, models_dir

def get_experiment_name(args):
    """Generate an experiment name if not provided."""
    if args.experiment_name:
        return args.experiment_name
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{args.dataset}_search_{timestamp}"

def setup_components(args):
    """Set up S-NAS components based on arguments."""
    # Set up dataset registry
    dataset_registry = DatasetRegistry(
        data_dir='./data',
        batch_size=args.batch_size,
        num_workers=4
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
    
    # Set up evaluator
    evaluator = Evaluator(
        dataset_registry=dataset_registry,
        model_builder=model_builder,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience
    )
    
    # Parse GPU IDs if provided
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    # Set up parallel evaluation if multiple GPUs specified
    if gpu_ids and len(gpu_ids) > 1:
        job_distributor = JobDistributor(
            num_workers=args.num_workers or len(gpu_ids),
            device_ids=gpu_ids
        )
        parallel_evaluator = ParallelEvaluator(
            evaluator=evaluator,
            job_distributor=job_distributor
        )
    else:
        job_distributor = None
        parallel_evaluator = None
    
    return {
        'dataset_registry': dataset_registry,
        'architecture_space': architecture_space,
        'model_builder': model_builder,
        'evaluator': evaluator,
        'job_distributor': job_distributor,
        'parallel_evaluator': parallel_evaluator
    }

def run_search(args, components, experiment_name, results_dir, models_dir):
    """Run the neural architecture search process."""
    logger.info(f"Starting search experiment: {experiment_name}")
    
    # Create evolutionary search
    search = EvolutionarySearch(
        architecture_space=components['architecture_space'],
        evaluator=components['evaluator'],
        dataset_name=args.dataset,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=3,
        metric='val_acc',
        save_history=True
    )
    
    # Initialize population
    logger.info("Initializing population...")
    search.initialize_population()
    
    # Run evolutionary search
    for generation in range(args.generations):
        logger.info(f"Generation {generation + 1}/{args.generations}")
        
        # Use fast mode for early generations
        use_fast_mode = generation < args.fast_mode_gens
        
        # Evaluate population
        if components['parallel_evaluator'] and not use_fast_mode:
            # Use parallel evaluation for regular evaluation
            logger.info("Using parallel evaluation")
            fitness_scores = components['parallel_evaluator'].evaluate_architectures(
                search.population, args.dataset, fast_mode=use_fast_mode
            )
            search.fitness_scores = fitness_scores
        else:
            # Use standard evaluation
            logger.info(f"Evaluating population (fast_mode={use_fast_mode})")
            search.evaluate_population(fast_mode=use_fast_mode)
        
        # Log generation statistics
        best_idx = search.fitness_scores.index(max(search.fitness_scores))
        best_fitness = search.fitness_scores[best_idx]
        avg_fitness = sum(search.fitness_scores) / len(search.fitness_scores)
        logger.info(f"Generation stats: Avg fitness: {avg_fitness:.4f}, Best fitness: {best_fitness:.4f}")
        
        # Create next generation (except for last iteration)
        if generation < args.generations - 1:
            search.population = search.create_next_generation()
    
    # Get best architecture and save results
    best_architecture = search.best_architecture
    best_fitness = search.best_fitness
    history = search.history
    
    # Save results
    save_results(experiment_name, history, best_architecture, best_fitness, results_dir, models_dir)
    
    logger.info(f"Search completed! Best fitness: {best_fitness:.4f}")
    return best_architecture, best_fitness, history

def evaluate_architecture(args, components, architecture_path, results_dir):
    """Evaluate a specific architecture defined in a JSON file."""
    logger.info(f"Evaluating architecture from: {architecture_path}")
    
    # Load architecture from JSON file
    with open(architecture_path, 'r') as f:
        architecture = json.load(f)
    
    # Get dataset configuration
    dataset_config = components['dataset_registry'].get_dataset_config(args.dataset)
    
    # Ensure architecture has required dataset-specific parameters
    if 'input_shape' not in architecture:
        architecture['input_shape'] = dataset_config['input_shape']
    if 'num_classes' not in architecture:
        architecture['num_classes'] = dataset_config['num_classes']
    
    # Evaluate the architecture
    evaluation = components['evaluator'].evaluate(
        architecture, args.dataset, fast_mode=False
    )
    
    # Log results
    logger.info(f"Evaluation results:")
    logger.info(f"  Test accuracy: {evaluation['test_acc']:.4f}")
    logger.info(f"  Best validation accuracy: {evaluation['best_val_acc']:.4f}")
    logger.info(f"  Epochs trained: {evaluation['epochs_trained']}")
    logger.info(f"  Training time: {evaluation['training_time']:.2f} seconds")
    
    # Save evaluation results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    eval_filename = f"{args.dataset}_eval_{timestamp}.json"
    eval_path = os.path.join(results_dir, eval_filename)
    
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {eval_path}")
    return evaluation

def save_results(experiment_name, history, best_architecture, best_fitness, results_dir, models_dir):
    """Save search results to disk."""
    # Save history as pickle
    history_path = os.path.join(results_dir, f"{experiment_name}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Save best architecture as JSON
    arch_path = os.path.join(results_dir, f"{experiment_name}_best.json")
    with open(arch_path, 'w') as f:
        json.dump(best_architecture, f, indent=2)
    
    # Save summary of results
    summary = {
        'experiment_name': experiment_name,
        'best_fitness': best_fitness,
        'architecture_depth': best_architecture['num_layers'],
        'architecture_path': arch_path,
        'history_path': history_path,
        'time_completed': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = os.path.join(results_dir, f"{experiment_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"  Best architecture: {arch_path}")
    logger.info(f"  History: {history_path}")
    logger.info(f"  Summary: {summary_path}")

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directories
    results_dir, models_dir = setup_output_dirs(args.output_dir)
    
    # Generate experiment name
    experiment_name = get_experiment_name(args)
    
    # Set up components
    components = setup_components(args)
    
    try:
        # Run in evaluation mode if an architecture file is provided
        if args.evaluate:
            evaluate_architecture(args, components, args.evaluate, results_dir)
        else:
            # Run search
            best_architecture, best_fitness, history = run_search(
                args, components, experiment_name, results_dir, models_dir
            )
            
            # Print summary of results
            print("\nSearch Results Summary:")
            print(f"  Dataset: {args.dataset}")
            print(f"  Best validation accuracy: {best_fitness:.4f}")
            print(f"  Architecture depth: {best_architecture['num_layers']} layers")
            print(f"  Results saved as: {experiment_name}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
    finally:
        # Clean up resources
        if components.get('job_distributor'):
            components['job_distributor'].stop()
        
        logger.info("Exiting S-NAS")

if __name__ == "__main__":
    main()