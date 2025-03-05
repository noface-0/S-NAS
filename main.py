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
from snas.search.enas_search import ENASSearch
from snas.search.pnas.pnas_search import PNASSearch
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
    parser = argparse.ArgumentParser(
        description='S-NAS: Simple Neural Architecture Search')

    # Search method options
    parser.add_argument('--search-method', type=str, default='evolutionary',
                       choices=['evolutionary', 'pnas', 'enas'],
                       help='Neural architecture search method to use')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'mnist',
                                 'kmnist', 'qmnist', 'emnist', 'fashion_mnist',
                                 'stl10', 'dtd', 'gtsrb'],
                        help='Dataset to use for architecture search')
    parser.add_argument('--custom-csv-dataset', type=str, default=None,
                        help='Path to CSV file for custom dataset')
    parser.add_argument('--custom-folder-dataset', type=str, default=None,
                        help='Path to folder for custom image dataset')
    parser.add_argument(
        '--custom-dataset-name',
        type=str,
        default='custom_dataset',
        help='Name for the custom dataset')
    parser.add_argument('--image-size', type=str, default='32x32',
                        choices=['32x32', '64x64', '224x224'],
                        help='Image size for custom dataset')

    # Network type options
    parser.add_argument(
        '--network-type',
        type=str,
        default=None,
        choices=[
            'all',
            'cnn',
            'mlp',
            'enhanced_mlp',
            'resnet',
            'mobilenet',
            'densenet',
            'shufflenetv2',
            'efficientnet'],
        help='Type of neural network architecture to search ('
        'all=search all types, None=use network_type in architecture file)')

    # Search parameters
    parser.add_argument('--population-size', type=int, default=20,
                        help='Population size for evolutionary search')
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of generations for evolutionary search')
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                        help='Mutation rate for evolutionary search')
    parser.add_argument(
        '--elite-size',
        type=int,
        default=2,
        help='Number of top architectures to preserve unchanged')
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=5,
        help='Save checkpoint every N generations (0 to disable)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to checkpoint file to resume search from')

    # Training parameters
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Maximum epochs per architecture evaluation')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument(
        '--min-delta',
        type=float,
        default=0.001,
        help='Minimum change to qualify as improvement for early stopping')
    parser.add_argument(
        '--monitor',
        type=str,
        default='val_acc',
        choices=[
            'val_acc',
            'val_loss'],
        help='Metric to monitor for early stopping')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of workers for data loading (default: CPU count)')
    parser.add_argument(
        '--fast-mode-gens',
        type=int,
        default=2,
        help='Number of generations to use fast evaluation mode')

    # Evaluation options
    parser.add_argument(
        '--extended-metrics',
        action='store_true',
        help='Compute extended metrics (precision, recall, F1, etc.)')

    # Parameter sharing is enabled by default, progressive search can be toggled
    parser.add_argument(
        '--enable-progressive',
        action='store_true',
        default=True,
        help='Enable progressive search (starting with simpler architectures)')
    parser.add_argument(
        '--weight-sharing-max-models',
        type=int,
        default=100,
        help='Maximum number of models to keep in the weight sharing pool')
        
    # PNAS specific parameters
    parser.add_argument('--beam-size', type=int, default=10,
                      help='Number of architectures to keep in the beam (PNAS)')
    parser.add_argument('--max-complexity', type=int, default=3,
                      help='Maximum complexity level to explore (PNAS)')
    parser.add_argument('--num-expansions', type=int, default=5,
                      help='Number of expansions per architecture in the beam (PNAS)')
                      
    # ENAS specific parameters
    parser.add_argument('--controller-sample-count', type=int, default=50,
                      help='Number of architectures to sample per iteration (ENAS)')
    parser.add_argument('--controller-entropy-weight', type=float, default=0.0001,
                      help='Weight for entropy term in controller update (ENAS)')
                      
    # PNAS+ENAS hybrid parameters
    parser.add_argument('--use-shared-weights', action='store_true',
                      help='Enable PNAS+ENAS hybrid mode with shared weights')
    parser.add_argument('--shared-weights-importance', type=float, default=0.5,
                      help='Balance between surrogate model and shared weights (0.0-1.0)')
    parser.add_argument('--dynamic-importance-weight', action='store_true',
                      help='Dynamically adjust importance weight based on surrogate accuracy')
    
    # Min and max layers parameters
    parser.add_argument('--min-layers', type=int, default=2, 
                      help='Minimum number of layers in the architecture')
    parser.add_argument('--max-layers', type=int, default=10,
                      help='Maximum number of layers in the architecture')

    # Export options
    parser.add_argument('--export-model', action='store_true',
                        help='Export the best model after search')
    parser.add_argument('--export-json', type=str, default=None,
                        help='Export the best architecture to a JSON file')
    parser.add_argument(
        '--export-format',
        type=str,
        default='torchscript',
        choices=[
            'torchscript',
            'onnx',
            'quantized',
            'mobile',
            'all'],
        help='Format to export the model to')
    parser.add_argument(
        '--export-dir',
        type=str,
        default='output/exported_models',
        help='Directory to save exported models')
    parser.add_argument(
        '--gpu-ids',
        type=str,
        default=None,
        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')

    # Output options
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save results')
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for the experiment (default: auto-generated)')

    # Evaluation mode
    parser.add_argument(
        '--evaluate',
        type=str,
        default=None,
        help='Path to architecture JSON file for evaluation only (no search)')

    # Device selection
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use for computation (e.g., "cuda:0", "cpu")')

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
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Register custom dataset if provided
    custom_dataset_used = False
    if args.custom_csv_dataset:
        logger.info(
            f"Registering custom CSV dataset from {args.custom_csv_dataset}")
        dataset_registry.register_csv_dataset(
            name=args.custom_dataset_name,
            csv_file=args.custom_csv_dataset,
            image_size=args.image_size
        )
        args.dataset = args.custom_dataset_name
        custom_dataset_used = True
    elif args.custom_folder_dataset:
        logger.info(
            f"Registering custom folder dataset from {args.custom_folder_dataset}")
        dataset_registry.register_folder_dataset(
            name=args.custom_dataset_name,
            root_dir=args.custom_folder_dataset,
            image_size=args.image_size
        )
        args.dataset = args.custom_dataset_name
        custom_dataset_used = True

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

    # Set up evaluator with weight sharing always enabled
    evaluator = Evaluator(
        dataset_registry=dataset_registry,
        model_builder=model_builder,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
        enable_weight_sharing=True,  # Always enable weight sharing
        weight_sharing_max_models=args.weight_sharing_max_models
    )

    # Setup GPU IDs if specified
    if args.gpu_ids:
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
    else:
        gpu_ids = None

    # Setup job distributor for parallel evaluation if multiple GPUs
    job_distributor = None
    parallel_evaluator = None

    if gpu_ids and len(gpu_ids) > 1:
        job_distributor = JobDistributor(
            num_workers=len(gpu_ids),
            device_ids=gpu_ids
        )
        parallel_evaluator = ParallelEvaluator(
            evaluator=evaluator,
            job_distributor=job_distributor
        )

    # Return all components
    return {
        'dataset_registry': dataset_registry,
        'architecture_space': architecture_space,
        'model_builder': model_builder,
        'evaluator': evaluator,
        'job_distributor': job_distributor,
        'parallel_evaluator': parallel_evaluator
    }


def export_best_model(architecture, components, results, args):
    """Export the best model to various formats."""
    try:
        from snas.model_exporter import ModelExporter
    except ImportError:
        logger.error("model_exporter module not found. Skipping model export.")
        return None

    # Create model exporter
    os.makedirs(args.export_dir, exist_ok=True)
    exporter = ModelExporter(output_dir=args.export_dir)

    # Rebuild the model
    model = components['model_builder'].build_model(architecture)

    # Get dataset configuration
    dataset_config = components['dataset_registry'].get_dataset_config(
        args.dataset)
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
                path = exporter.export_to_torchscript(
                    model, input_shape, model_name)
            elif args.export_format == 'onnx':
                path = exporter.export_to_onnx(model, input_shape, model_name)
            elif args.export_format == 'quantized':
                path = exporter.export_quantized_model(
                    model, input_shape, model_name)
            elif args.export_format == 'mobile':
                path = exporter.export_model_for_mobile(
                    model, input_shape, model_name)

            export_paths[args.export_format] = path

            # Generate and save example code
            example_code = exporter.generate_example_code(
                args.export_format, path, input_shape
            )
            example_path = os.path.join(
                args.export_dir,
                f"{model_name}_{args.export_format}_example.py")
            with open(example_path, 'w') as f:
                f.write(example_code)

            logger.info(f"Model exported to {args.export_format} at: {path}")
            logger.info(f"Example code saved to: {example_path}")

        except Exception as e:
            logger.error(f"Error exporting model to {args.export_format}: {e}")

    # Create and save model report
    if args.extended_metrics:
        try:
            report = components['evaluator'].generate_model_report(
                model, architecture, results, args.export_dir
            )
            logger.info(
                f"Model report saved to: {report.get('report_path', 'unknown')}")
        except Exception as e:
            logger.error(f"Error generating model report: {e}")

    return export_paths


def run_search(args, components, experiment_name, results_dir, models_dir):
    """Run the neural architecture search process."""
    logger.info(f"Starting search experiment: {experiment_name}")

    # Select the appropriate search method based on user input
    if args.search_method == 'evolutionary':
        # Create evolutionary search with user-selected progressive search setting
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
            save_history=True,
            checkpoint_frequency=args.checkpoint_frequency,
            output_dir=args.output_dir,
            results_dir=results_dir,
            enable_progressive=args.enable_progressive
        )
    elif args.search_method == 'pnas':
        # Create PNAS search
        search = PNASSearch(
            architecture_space=components['architecture_space'],
            evaluator=components['evaluator'],
            dataset_name=args.dataset,
            beam_size=args.beam_size,
            num_expansions=args.num_expansions,
            max_complexity=args.max_complexity,
            metric='val_acc',
            checkpoint_frequency=args.checkpoint_frequency,
            output_dir=args.output_dir,
            results_dir=results_dir,
            use_shared_weights=args.use_shared_weights,
            shared_weights_importance=args.shared_weights_importance,
            dynamic_importance_weight=args.dynamic_importance_weight
        )
    elif args.search_method == 'enas':
        # Create ENAS search
        search = ENASSearch(
            architecture_space=components['architecture_space'],
            evaluator=components['evaluator'],
            dataset_name=args.dataset,
            controller_sample_count=args.controller_sample_count,
            controller_entropy_weight=args.controller_entropy_weight,
            weight_sharing_max_models=args.weight_sharing_max_models,
            metric='val_acc',
            checkpoint_frequency=args.checkpoint_frequency,
            output_dir=args.output_dir,
            results_dir=results_dir
        )
    else:
        # Default to evolutionary search as fallback
        logger.warning(f"Unknown search method: {args.search_method}. Using evolutionary search.")
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
            save_history=True,
            checkpoint_frequency=args.checkpoint_frequency,
            output_dir=args.output_dir,
            results_dir=results_dir,
            enable_progressive=args.enable_progressive
        )

    # Force specific network type if specified
    if args.network_type == 'all':
        # Let it search all network types
        logger.info("Searching all network architecture types")
    elif args.network_type is not None:
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
        
    # Apply min-layers and max-layers constraints if specified
    if (args.min_layers is not None or args.max_layers is not None) and hasattr(search.architecture_space, 'set_layer_constraints'):
        logger.info(f"Setting layer constraints: min_layers={args.min_layers}, max_layers={args.max_layers}")
        search.architecture_space.set_layer_constraints(args.min_layers, args.max_layers)

    # Check if we should resume from checkpoint
    start_generation = 0
    if args.resume_from:
        try:
            logger.info(f"Attempting to resume from checkpoint: {args.resume_from}")
            
            # Different search methods have different methods for running the search
            if args.search_method == 'evolutionary':
                # Call the evolve method with the resume_from parameter
                best_architecture, best_fitness, history = search.evolve(
                    fast_mode_generations=args.fast_mode_gens,
                    resume_from=args.resume_from
                )
            elif args.search_method == 'pnas' or args.search_method == 'enas':
                # Call the search method with the resume_from parameter
                best_architecture, best_fitness, history = search.search(
                    resume_from=args.resume_from
                )
            else:
                # Fall back to evolutionary search
                best_architecture, best_fitness, history = search.evolve(
                    fast_mode_generations=args.fast_mode_gens,
                    resume_from=args.resume_from
                )
                
            # Save results and return early since the search method handles the full search process
            save_results(
                experiment_name,
                history,
                best_architecture,
                best_fitness,
                results_dir,
                models_dir,
                args=args)
            logger.info(f"Search resumed and completed! Best fitness: {best_fitness:.4f}")
            return best_architecture, best_fitness, history
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            logger.info("Starting new search instead")

    # Run different search algorithms depending on the method
    if args.search_method == 'evolutionary':
        # Initialize population
        logger.info("Initializing population...")
        search.initialize_population()

        # Run evolutionary search
        for generation in range(start_generation, args.generations):
            logger.info(f"Generation {generation + 1}/{args.generations}")

            # Use fast mode for early generations
            use_fast_mode = generation < args.fast_mode_gens

            # Evaluate population
            if components['parallel_evaluator'] and not use_fast_mode:
                # Use parallel evaluation for regular evaluation
                logger.info("Using parallel evaluation")
                fitness_scores = components['parallel_evaluator'].evaluate_architectures(
                    search.population, args.dataset, fast_mode=use_fast_mode)
                search.fitness_scores = fitness_scores
            else:
                # Use standard evaluation
                logger.info(f"Evaluating population (fast_mode={use_fast_mode})")
                search.evaluate_population(fast_mode=use_fast_mode)

            # Log generation statistics
            best_idx = search.fitness_scores.index(max(search.fitness_scores))
            best_fitness = search.fitness_scores[best_idx]
            avg_fitness = sum(search.fitness_scores) / len(search.fitness_scores)
            logger.info(
                f"Generation stats: Avg fitness: {avg_fitness:.4f}, Best fitness: {best_fitness:.4f}")

            # Create next generation (except for last iteration)
            if generation < args.generations - 1:
                search.population = search.create_next_generation()

        # Get best architecture and save results
        best_architecture = search.best_architecture
        best_fitness = search.best_fitness
        history = search.history
        
    elif args.search_method == 'pnas' or args.search_method == 'enas':
        # Run PNAS or ENAS search
        logger.info(f"Running {args.search_method.upper()} search...")
        best_architecture, best_fitness, history = search.search(
            progress_callback=None  # Can implement a progress callback function if needed
        )

    # Save results
    save_results(
        experiment_name,
        history,
        best_architecture,
        best_fitness,
        results_dir,
        models_dir,
        args=args)

    # Export architecture to JSON if requested
    if args.export_json:
        logger.info(f"Exporting best architecture to JSON: {args.export_json}")
        os.makedirs(os.path.dirname(os.path.abspath(args.export_json)), exist_ok=True)
        with open(args.export_json, 'w') as f:
            json.dump(best_architecture, f, indent=2)
        logger.info(f"Architecture exported successfully to: {args.export_json}")

    # Export model if requested
    if args.export_model:
        # Perform final evaluation to get complete metrics
        if args.extended_metrics:
            logger.info("Performing final evaluation with extended metrics")
            results = components['evaluator'].evaluate(
                best_architecture, args.dataset, fast_mode=False)
        else:
            # Add required fields for export
            results = {
                'test_acc': best_fitness,
                'architecture': best_architecture,
                'dataset': args.dataset
            }
            
            # Add additional fields based on search method
            if args.search_method == 'pnas' and args.use_shared_weights:
                results['search_method'] = 'pnas+enas_hybrid'
                results['shared_weights_importance'] = args.shared_weights_importance
                results['dynamic_importance_weight'] = args.dynamic_importance_weight
            else:
                results['search_method'] = args.search_method

        # Export the model
        logger.info(f"Exporting best model to {args.export_format} format")
        export_paths = export_best_model(
            best_architecture, components, results, args)

        if export_paths:
            logger.info(
                f"Model exported successfully to: {', '.join(export_paths.keys())}")

    logger.info(f"Search completed! Best fitness: {best_fitness:.4f}")
    return best_architecture, best_fitness, history


def evaluate_architecture(args, components, architecture_path, results_dir):
    """Evaluate a specific architecture defined in a JSON file."""
    logger.info(f"Evaluating architecture from: {architecture_path}")

    # Load architecture from JSON file
    with open(architecture_path, 'r') as f:
        architecture = json.load(f)

    # Get dataset configuration
    dataset_config = components['dataset_registry'].get_dataset_config(
        args.dataset)

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
    logger.info(
        f"  Best validation accuracy: {evaluation['best_val_acc']:.4f}")
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


def save_results(
        experiment_name,
        history,
        best_architecture,
        best_fitness,
        results_dir,
        models_dir,
        args=None):
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
        'time_completed': time.strftime("%Y-%m-%d %H:%M:%S"),
        'search_method': args.search_method if args else 'unknown'
    }
    
    # Add method-specific details if available
    if args:
        if args.search_method == 'pnas':
            summary['beam_size'] = args.beam_size
            summary['max_complexity'] = args.max_complexity
            
            if args.use_shared_weights:
                summary['hybrid_mode'] = True
                summary['shared_weights_importance'] = args.shared_weights_importance
                summary['dynamic_importance_weight'] = args.dynamic_importance_weight
        
        elif args.search_method == 'enas':
            summary['controller_sample_count'] = args.controller_sample_count
        
        elif args.search_method == 'evolutionary':
            summary['population_size'] = args.population_size
            summary['generations'] = args.generations
            summary['mutation_rate'] = args.mutation_rate

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
            print(f"  Search method: {args.search_method}")
            print(f"  Dataset: {args.dataset}")
            print(f"  Best validation accuracy: {best_fitness:.4f}")
            print(f"  Architecture depth: {best_architecture['num_layers']} layers")
            print(f"  Network type: {best_architecture.get('network_type', 'cnn')}")
            print(f"  Results saved as: {experiment_name}")
            
            # Print additional details based on search method
            if args.search_method == 'pnas' and args.use_shared_weights:
                print(f"  PNAS+ENAS hybrid mode enabled")
                print(f"  Shared weights importance: {args.shared_weights_importance}")
                if args.dynamic_importance_weight:
                    print(f"  Dynamic importance weight adjustment enabled")
            elif args.search_method == 'enas':
                print(f"  Controller sample count: {args.controller_sample_count}")
            elif args.search_method == 'evolutionary':
                print(f"  Population size: {args.population_size}")
                print(f"  Generations: {args.generations}")

            if args.export_model:
                print(f"  Model exported to: {args.export_dir}")
                
            if args.export_json:
                print(f"  Architecture exported to: {args.export_json}")

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
