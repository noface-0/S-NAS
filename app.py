import os
import time
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import multiprocessing
import logging

# Import pandas with error handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available. Some visualizations will be limited.")

# Import torch with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import S-NAS components
from snas.data.dataset_registry import DatasetRegistry
from snas.architecture.architecture_space import ArchitectureSpace
from snas.architecture.model_builder import ModelBuilder
from snas.search.evaluator import Evaluator
from snas.search.evolutionary_search import EvolutionarySearch
from snas.utils.job_distributor import JobDistributor, ParallelEvaluator
from snas.visualization.visualizer import SearchVisualizer

# Constants
OUTPUT_DIR = "output"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Utility functions ----
def save_search_results(dataset_name, search_history, best_architecture, best_fitness):
    """Save search results to disk."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{dataset_name}_search_{timestamp}"
    
    # Save history
    with open(os.path.join(RESULTS_DIR, f"{filename}_history.pkl"), 'wb') as f:
        pickle.dump(search_history, f)
    
    # Save best architecture as JSON
    with open(os.path.join(RESULTS_DIR, f"{filename}_best.json"), 'w') as f:
        json.dump(best_architecture, f, indent=2)
        
    return filename

def load_search_results(filename):
    """Load search results from disk with error handling."""
    try:
        # Load history
        history_path = os.path.join(RESULTS_DIR, f"{filename}_history.pkl")
        if not os.path.exists(history_path):
            raise FileNotFoundError(f"History file not found: {history_path}")
            
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        # History should be a dictionary
        if not isinstance(history, dict):
            raise ValueError(f"Invalid history format: expected dictionary, got {type(history)}")
        
        # Load best architecture
        arch_path = os.path.join(RESULTS_DIR, f"{filename}_best.json")
        if not os.path.exists(arch_path):
            raise FileNotFoundError(f"Architecture file not found: {arch_path}")
            
        with open(arch_path, 'r') as f:
            best_architecture = json.load(f)
        
        # Architecture should be a dictionary
        if not isinstance(best_architecture, dict):
            raise ValueError(f"Invalid architecture format: expected dictionary, got {type(best_architecture)}")
            
        return history, best_architecture
    except (FileNotFoundError, json.JSONDecodeError, pickle.UnpicklingError) as e:
        # Re-raise with more context
        raise Exception(f"Failed to load search results: {str(e)}")
    except Exception as e:
        # Catch-all for unexpected errors
        raise Exception(f"Unexpected error loading search results: {str(e)}")

@st.cache_data
def get_available_datasets():
    """Get list of available datasets without loading them."""
    return [
        "cifar10", "cifar100", "svhn", "mnist", 
        "kmnist", "qmnist", "emnist", "fashion_mnist",
        "stl10", "dtd", "gtsrb"
    ]

@st.cache_resource
def get_network_types():
    """Get list of available network types."""
    return ["all", "cnn", "mlp", "enhanced_mlp", "resnet", "mobilenet", "densenet", "shufflenetv2", "efficientnet"]

def get_result_files():
    """Get list of available result files."""
    return [f.split("_history.pkl")[0] for f in os.listdir(RESULTS_DIR) 
           if f.endswith("_history.pkl")]

def initialize_components(dataset_name, batch_size, num_workers, pin_memory, _gpu_ids, device, 
                          custom_dataset_params=None):
    """Initialize components for a specific dataset."""
    logger.info(f"Initializing components for dataset: {dataset_name}")
    
    try:
        # Create dataset registry (but don't load datasets yet)
        dataset_registry = DatasetRegistry(
            data_dir='./data',
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Handle custom datasets
        if custom_dataset_params and dataset_name.startswith("custom_"):
            logger.info(f"Setting up custom dataset: {dataset_name}")
            
            try:
                # Parse image size
                image_size = custom_dataset_params.get('image_size', '224x224')
                width, height = map(int, image_size.split('x'))
                
                if custom_dataset_params['type'] == 'folder':
                    # Register folder-based dataset
                    folder_path = custom_dataset_params['folder_path']
                    logger.info(f"Registering folder dataset from: {folder_path}")
                    
                    dataset_config = dataset_registry.register_folder_dataset(
                        folder_path=folder_path,
                        dataset_name=dataset_name,
                        image_size=(height, width)
                    )
                else:  # CSV file
                    # Register CSV-based dataset
                    csv_path = custom_dataset_params['csv_path']
                    image_col = custom_dataset_params['image_column']
                    label_col = custom_dataset_params['label_column']
                    
                    logger.info(f"Registering CSV dataset from: {csv_path}")
                    logger.info(f"  Image column: {image_col}, Label column: {label_col}")
                    
                    dataset_config = dataset_registry.register_csv_dataset(
                        csv_path=csv_path,
                        dataset_name=dataset_name,
                        image_column=image_col,
                        label_column=label_col,
                        image_size=(height, width)
                    )
                    
                logger.info(f"Successfully registered custom dataset with {dataset_config['num_classes']} classes")
                
            except Exception as e:
                # Provide detailed error message for custom dataset setup
                error_msg = f"Error setting up custom dataset: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                
                # Fall back to a default dataset
                dataset_name = "mnist"  # Fallback to a simple dataset
                st.warning(f"Falling back to {dataset_name} dataset due to error in custom dataset setup")
                dataset_config = dataset_registry.get_dataset_config(dataset_name)
        else:
            # Get configuration for built-in dataset
            logger.info(f"Using built-in dataset: {dataset_name}")
            dataset_config = dataset_registry.get_dataset_config(dataset_name)
        
        # Create architecture space
        architecture_space = ArchitectureSpace(
            input_shape=dataset_config["input_shape"],
            num_classes=dataset_config["num_classes"]
        )
        
        # Create model builder
        model_builder = ModelBuilder(device=device)
        
        return {
            'dataset_registry': dataset_registry,
            'architecture_space': architecture_space,
            'model_builder': model_builder,
            'dataset_config': dataset_config
        }
        
    except Exception as e:
        # Comprehensive error handling
        error_msg = f"Error initializing components: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        
        # Raise the exception to propagate it
        raise

def main():
    """Main function containing all Streamlit UI code."""
    # Set page configuration
    st.set_page_config(page_title="S-NAS: Simple Neural Architecture Search", layout="wide")

    # App title and introduction
    st.title("S-NAS: Simple Neural Architecture Search")

    st.markdown("""
    This application provides an interface for the S-NAS system, which automates the discovery
    of optimal neural network architectures for specific datasets. Rather than manually designing
    neural networks, S-NAS efficiently explores different architecture configurations to find
    ones that perform best on predefined benchmark datasets.
    
    S-NAS incorporates two advanced efficiency techniques:
    * **Parameter Sharing** (from ENAS paper): Reuses weights between similar architectures (always enabled)
    * **Progressive Search** (from PNAS paper): Gradually increases architecture complexity (can be toggled)
    
    ### Note on Progressive Search and MLP Bias
    When using progressive search with simpler datasets like MNIST, there may be a bias toward MLP architectures
    in the search results. If you want to explore more CNN architectures on simpler datasets, consider:
    * Disabling progressive search using the checkbox in the sidebar
    * Explicitly selecting a network type (like CNN) from the dropdown
    """)

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Dataset selection group
    dataset_tab1, dataset_tab2 = st.sidebar.tabs(["Built-in Datasets", "Custom Dataset"])
    
    with dataset_tab1:
        # Built-in dataset selection
        dataset_options = get_available_datasets()
        selected_dataset = st.selectbox("Select Dataset", dataset_options)
        use_custom_dataset = False
    
    with dataset_tab2:
        # Custom dataset options
        st.header("Custom Dataset")
        custom_dataset_type = st.radio("Dataset Type", ["Folder Structure", "CSV File"])
        
        if custom_dataset_type == "Folder Structure":
            custom_folder_path = st.text_input("Dataset Folder Path", 
                                              help="Path to a folder with class subfolders. Each subfolder should contain images of that class.")
            custom_dataset_name = st.text_input("Dataset Name", "custom_dataset")
            
            # Preview button
            if st.button("Validate Folder Structure"):
                if os.path.exists(custom_folder_path):
                    subfolders = [f for f in os.listdir(custom_folder_path) 
                                 if os.path.isdir(os.path.join(custom_folder_path, f))]
                    if subfolders:
                        st.success(f"Found {len(subfolders)} class folders: {', '.join(subfolders[:5])}" + 
                                 ("..." if len(subfolders) > 5 else ""))
                    else:
                        st.error("No class subfolders found in the specified directory.")
                else:
                    st.error("The specified folder path does not exist.")
                    
            use_custom_dataset = st.checkbox("Use This Custom Dataset")
            
        else:  # CSV File
            custom_csv_path = st.text_input("CSV File Path",
                                          help="Path to a CSV file with image paths and labels.")
            custom_dataset_name = st.text_input("Dataset Name", "custom_dataset")
            image_column = st.text_input("Image Column Name", "image")
            label_column = st.text_input("Label Column Name", "label")
            
            # Preview button
            if st.button("Validate CSV File"):
                if os.path.exists(custom_csv_path) and custom_csv_path.endswith('.csv'):
                    try:
                        import pandas as pd
                        df = pd.read_csv(custom_csv_path)
                        if image_column in df.columns and label_column in df.columns:
                            unique_labels = df[label_column].unique()
                            st.success(f"Valid CSV file. Found {len(unique_labels)} unique classes.")
                            st.write(f"Sample data: {df.head(3)}")
                        else:
                            missing_cols = []
                            if image_column not in df.columns:
                                missing_cols.append(image_column)
                            if label_column not in df.columns:
                                missing_cols.append(label_column)
                            st.error(f"Column(s) not found in CSV: {', '.join(missing_cols)}")
                            st.write(f"Available columns: {', '.join(df.columns)}")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                else:
                    st.error("The specified file does not exist or is not a CSV file.")
                    
            use_custom_dataset = st.checkbox("Use This Custom Dataset")
        
        # Common custom dataset settings
        image_size = st.text_input("Image Size (WxH)", "224x224", 
                                 help="Format: widthxheight, e.g., 224x224")
        
    # Determine which dataset to use
    if use_custom_dataset:
        # Use custom dataset
        if custom_dataset_type == "Folder Structure":
            if not os.path.exists(custom_folder_path):
                st.sidebar.error("The specified folder path does not exist!")
                selected_dataset = dataset_options[0]  # Fallback to default
            else:
                selected_dataset = "custom_" + custom_dataset_name
                # We'll handle this case specially when initializing components
        else:  # CSV File
            if not os.path.exists(custom_csv_path) or not custom_csv_path.endswith('.csv'):
                st.sidebar.error("The specified CSV file does not exist or is not a CSV file!")
                selected_dataset = dataset_options[0]  # Fallback to default
            else:
                selected_dataset = "custom_" + custom_dataset_name
                # We'll handle this case specially when initializing components

    # Hardware configuration
    st.sidebar.subheader("Hardware")
    # Check if PyTorch CUDA is available with error handling
    cuda_available = False
    if TORCH_AVAILABLE:
        try:
            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        
    use_gpu = st.sidebar.checkbox("Use GPU", value=cuda_available)
    if use_gpu and cuda_available and TORCH_AVAILABLE:
        try:
            num_gpus = torch.cuda.device_count()
            gpu_ids = st.sidebar.multiselect(
                "Select GPUs", 
                options=list(range(num_gpus)),
                default=list(range(min(num_gpus, 2)))
            )
        except Exception as e:
            st.sidebar.warning(f"Error accessing CUDA: {str(e)}. Falling back to CPU.")
            gpu_ids = []
    else:
        gpu_ids = []

    # Search parameters
    st.sidebar.subheader("Search Parameters")
    population_size = st.sidebar.slider("Population Size", 5, 50, 20)
    num_generations = st.sidebar.slider("Number of Generations", 5, 50, 10)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.1, 0.5, 0.2, 0.05)
    fast_mode_gens = st.sidebar.slider("Fast Evaluation Generations", 0, 5, 2)
    enable_progressive = st.sidebar.checkbox("Enable Progressive Search", value=True, 
                        help="Start with simpler architectures and increase complexity over generations. Disable to avoid MLP bias on simple datasets like MNIST.")
    
    # Checkpoint parameters (new)
    st.sidebar.subheader("Checkpoint Management")
    checkpoint_freq = st.sidebar.slider("Checkpoint Frequency (generations)", 0, 10, 2, 
                                      help="Number of generations between checkpoints. Set to 0 to disable.")
    
    # List available checkpoints
    checkpoint_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.pkl') and 'checkpoint' in f]
    checkpoint_options = ["None"] + checkpoint_files
    selected_checkpoint = st.sidebar.selectbox(
        "Resume From Checkpoint", 
        options=checkpoint_options,
        help="Select a checkpoint file to resume search from"
    )

    # Network type selection
    st.sidebar.subheader("Network Architecture")
    network_type_options = get_network_types()
    network_type = st.sidebar.selectbox(
        "Network Type", 
        options=network_type_options,
        index=0, 
        help="Type of neural network to search for"
    )

    # Training parameters
    st.sidebar.subheader("Training Parameters")
    max_epochs = st.sidebar.slider("Max Epochs per Evaluation", 5, 50, 10)
    batch_size = st.sidebar.select_slider(
        "Batch Size", 
        options=[32, 64, 128, 256, 512], 
        value=128
    )

    # Early stopping parameters
    patience = st.sidebar.slider("Early Stopping Patience", 1, 10, 3)
    min_delta = st.sidebar.slider("Min Improvement Delta", 0.0001, 0.01, 0.001, 0.0001)
    monitor = st.sidebar.radio("Monitor Metric", ["val_acc", "val_loss"], index=0)

    # Data loading parameters
    cpu_count = multiprocessing.cpu_count()
    num_workers = st.sidebar.slider("Data Loading Workers", 0, max(16, cpu_count), min(4, cpu_count))

    # View previous results
    st.sidebar.subheader("Previous Results")
    result_files = ["None"] + get_result_files()
    selected_result = st.sidebar.selectbox(
        "Select Previous Result", 
        options=result_files
    )

    # Determine device
    if use_gpu and cuda_available and TORCH_AVAILABLE and gpu_ids:
        device = f"cuda:{gpu_ids[0]}"
    else:
        device = "cpu"

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Search", "Results", "Export"])

    # Search tab
    with tab1:
        st.header("Neural Architecture Search")
        
        st.markdown(f"""
        ### Current Configuration
        - **Dataset**: {selected_dataset}
        - **Network Type**: {network_type}
        - **Population Size**: {population_size}
        - **Generations**: {num_generations}
        - **Hardware**: {"GPU" if use_gpu else "CPU"}
        """)
        
        # Status indicators for errors
        error_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Display custom dataset info if using one
        if use_custom_dataset and selected_dataset.startswith("custom_"):
            if custom_dataset_type == "Folder Structure":
                info_placeholder.info(f"Using custom dataset from folder: {custom_folder_path}")
            else:
                info_placeholder.info(f"Using custom dataset from CSV: {custom_csv_path}")
            
        # Display checkpoint info if using one
        if selected_checkpoint != "None":
            info_placeholder.info(f"Will resume search from checkpoint: {selected_checkpoint}")
        
        # Start search button
        if st.button("Start Search"):
            # Clear any previous messages
            error_placeholder.empty()
            info_placeholder.empty()
            
            with st.spinner("Initializing components..."):
                # Prepare custom dataset parameters if needed
                custom_dataset_params = None
                if use_custom_dataset and selected_dataset.startswith("custom_"):
                    if custom_dataset_type == "Folder Structure":
                        custom_dataset_params = {
                            'type': 'folder',
                            'folder_path': custom_folder_path,
                            'image_size': image_size
                        }
                    else:  # CSV File
                        custom_dataset_params = {
                            'type': 'csv',
                            'csv_path': custom_csv_path,
                            'image_column': image_column,
                            'label_column': label_column,
                            'image_size': image_size
                        }
                
                try:
                    # Initialize components only when needed
                    components = initialize_components(
                        selected_dataset, 
                        batch_size, 
                        num_workers, 
                        use_gpu and torch.cuda.is_available(),
                        gpu_ids,
                        device,
                        custom_dataset_params
                    )
                except Exception as e:
                    # Display error and exit
                    error_placeholder.error(f"Failed to initialize search components: {str(e)}")
                    return
                
                # Create evaluator with parameter sharing enabled by default
                evaluator = Evaluator(
                    dataset_registry=components['dataset_registry'],
                    model_builder=components['model_builder'],
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    monitor=monitor,
                    enable_weight_sharing=True,  # Parameter sharing always enabled
                    weight_sharing_max_models=100  # Keep up to 100 models in the sharing pool
                )
                
                # Set up job distributor if using multiple GPUs
                if use_gpu and len(gpu_ids) > 1:
                    job_distributor = JobDistributor(
                        num_workers=len(gpu_ids),
                        device_ids=gpu_ids
                    )
                    parallel_evaluator = ParallelEvaluator(
                        evaluator=evaluator,
                        job_distributor=job_distributor
                    )
                else:
                    parallel_evaluator = None
                
                # Check if resuming from checkpoint
                if selected_checkpoint != "None":
                    checkpoint_path = os.path.join(RESULTS_DIR, selected_checkpoint)
                    st.info(f"Resuming search from checkpoint: {selected_checkpoint}")
                    
                    # Load checkpoint data
                    with open(checkpoint_path, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                    
                    # Create search object with checkpoint data
                    search = EvolutionarySearch(
                        architecture_space=components['architecture_space'],
                        evaluator=evaluator,
                        dataset_name=selected_dataset,
                        population_size=population_size,
                        mutation_rate=mutation_rate,
                        generations=num_generations,
                        elite_size=2,
                        tournament_size=3,
                        metric=monitor,
                        save_history=True,
                        enable_progressive=enable_progressive,
                        checkpoint_data=checkpoint_data  # Provide checkpoint data
                    )
                else:
                    # Create new evolutionary search with user-selected progressive search setting
                    search = EvolutionarySearch(
                        architecture_space=components['architecture_space'],
                        evaluator=evaluator,
                        dataset_name=selected_dataset,
                        population_size=population_size,
                        mutation_rate=mutation_rate,
                        generations=num_generations,
                        elite_size=2,
                        tournament_size=3,
                        metric=monitor,
                        save_history=True,
                        enable_progressive=enable_progressive,  # User-controlled progressive search
                        checkpoint_frequency=checkpoint_freq  # Save checkpoints periodically
                    )
                
                # Force specific network type if not "all"
                if network_type != "all":
                    # Override the network_type in the sample_random_architecture method
                    original_sample = search.architecture_space.sample_random_architecture
                    
                    # Create wrapped function that forces network_type
                    def sample_with_fixed_type():
                        arch = original_sample()
                        arch['network_type'] = network_type
                        return arch
                    
                    # Replace the method
                    search.architecture_space.sample_random_architecture = sample_with_fixed_type
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run search with progress updates
            status_text.text("Initializing population...")
            search.initialize_population()
            
            for generation in range(num_generations):
                # Update status and progress
                progress = (generation + 1) / num_generations
                progress_bar.progress(progress)
                status_text.text(f"Generation {generation + 1}/{num_generations}")
                
                # For progressive search, check if complexity should be increased
                if search.enable_progressive:
                    # Update complexity level at transition points
                    transition_point = (search.complexity_level * num_generations) // (search.max_complexity_level + 1)
                    if generation >= transition_point and search.complexity_level < search.max_complexity_level:
                        search.complexity_level += 1
                        st.text(f"Increasing architecture complexity to level {search.complexity_level}")
                    
                    # Add current complexity level to history
                    if search.save_history:
                        search.history['complexity_level'].append(search.complexity_level)
                
                # Save checkpoint if needed
                if checkpoint_freq > 0 and generation > 0 and generation % checkpoint_freq == 0:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    checkpoint_filename = f"{selected_dataset}_checkpoint_gen{generation}_{timestamp}.pkl"
                    checkpoint_path = os.path.join(RESULTS_DIR, checkpoint_filename)
                    
                    # Create checkpoint data
                    checkpoint_data = {
                        'population': search.population,
                        'fitness_scores': search.fitness_scores,
                        'best_architecture': search.best_architecture if hasattr(search, 'best_architecture') else None,
                        'best_fitness': search.best_fitness if hasattr(search, 'best_fitness') else None,
                        'history': search.history,
                        'generation': generation,
                        'complexity_level': search.complexity_level,
                        'dataset_name': selected_dataset
                    }
                    
                    # Save checkpoint
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    
                    status_text.text(f"Checkpoint saved at generation {generation} to {checkpoint_filename}")
                
                # Use fast mode for early generations
                use_fast_mode = generation < fast_mode_gens
                
                # Evaluate population
                if parallel_evaluator and not use_fast_mode:
                    # Use parallel evaluation for regular evaluation
                    fitness_scores = parallel_evaluator.evaluate_architectures(
                        search.population, selected_dataset, fast_mode=use_fast_mode
                    )
                    search.fitness_scores = fitness_scores
                else:
                    # Use standard evaluation
                    search.evaluate_population(fast_mode=use_fast_mode)
                
                # Record statistics for this generation
                if len(search.fitness_scores) > 0:
                    if search.higher_is_better:
                        best_fitness = max(search.fitness_scores)
                        best_idx = search.fitness_scores.index(best_fitness)
                    else:
                        best_fitness = min(search.fitness_scores)
                        best_idx = search.fitness_scores.index(best_fitness)
                    
                    avg_fitness = sum(search.fitness_scores) / len(search.fitness_scores)
                    best_arch = search.population[best_idx].copy()
                    diversity = search.calculate_diversity()
                    
                    # Save history
                    if search.save_history:
                        search.history['generations'].append(generation)
                        search.history['best_fitness'].append(best_fitness)
                        search.history['avg_fitness'].append(avg_fitness)
                        search.history['best_architecture'].append(best_arch)
                        search.history['population_diversity'].append(diversity)
                        search.history['evaluation_times'].append(0)  # Placeholder for evaluation times
                
                # Create next generation (except for last iteration)
                if generation < num_generations - 1:
                    search.population = search.create_next_generation()
            
            # Get best architecture and save results
            best_architecture = search.best_architecture
            best_fitness = search.best_fitness
            history = search.history
            
            # Save results
            result_filename = save_search_results(
                selected_dataset, history, best_architecture, best_fitness
            )
            
            # Update status
            progress_bar.progress(1.0)
            status_text.text(f"Search completed! Best fitness: {best_fitness:.4f}")
            
            # Show summary
            st.success(f"""
            ### Search Completed
            - Best validation accuracy: {best_fitness:.4f}
            - Architecture depth: {best_architecture['num_layers']} layers
            - Results saved as: {result_filename}
            """)
            
            # Create visualizer and show plots
            visualizer = SearchVisualizer(components['architecture_space'])
            
            # Plot search progress
            st.subheader("Search Progress")
            fig = visualizer.plot_search_progress(history, metric="multiple")
            st.pyplot(fig)
            
            # Plot best architecture
            st.subheader("Best Architecture")
            
            # Create tabs for different visualization types
            vis_tabs = st.tabs(["Network Graph", "PyTorch Model"])
            
            with vis_tabs[0]:
                fig = visualizer.visualize_architecture_networks([best_architecture], ["Best Architecture"])
                st.pyplot(fig)
                
            with vis_tabs[1]:
                # Try model visualization with torchviz
                torch_fig = visualizer.visualize_torch_model(best_architecture, components['model_builder'])
                st.pyplot(torch_fig)
                
                # Add model summary
                st.subheader("Model Summary")
                try:
                    if TORCH_AVAILABLE:
                        model = components['model_builder'].build_model(best_architecture)
                        
                        # Count parameters
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        
                        # Display summary
                        st.markdown(f"""
                        - **Total Parameters**: {total_params:,}
                        - **Trainable Parameters**: {trainable_params:,}
                        """)
                        
                        # Display model structure
                        st.code(str(model))
                    else:
                        st.warning("PyTorch is not available. Cannot display model summary.")
                except Exception as e:
                    st.warning(f"Could not generate model summary: {e}")

    # Results tab
    with tab2:
        st.header("Search Results")
        
        if selected_result != "None":
            # Load selected results
            with st.spinner("Loading results..."):
                try:
                    history, best_architecture = load_search_results(selected_result)
                    
                    # Handle case where history might be incomplete or empty
                    if 'best_fitness' not in history or not history['best_fitness']:
                        st.error("The selected result file appears to be incomplete or corrupted.")
                        best_fitness = 0.0
                    else:
                        best_fitness = max(history['best_fitness'])
                    
                    # Create visualizer
                    visualizer = SearchVisualizer()
                    
                    # Add debug button
                    if st.checkbox("Debug History Data"):
                        st.subheader("History Data Debug")
                        debug_fig = visualizer.debug_history_data(history)
                        st.pyplot(debug_fig)
                        
                except Exception as e:
                    st.error(f"Error loading results: {str(e)}")
                    st.info("The selected result file might be corrupted or in an incompatible format.")
                    return
            
            # Display summary
            try:
                # Check if architecture is valid before attempting to display details
                if not isinstance(best_architecture, dict) or 'num_layers' not in best_architecture:
                    st.error("The architecture information in this result file is incomplete or invalid.")
                    return
                
                # Try to extract dataset from filename, with a fallback option
                try:
                    dataset_name = selected_result.split("_")[0]
                except:
                    dataset_name = "unknown"
                
                st.markdown(f"""
                ### Result Summary: {selected_result}
                - **Dataset**: {dataset_name}
                - **Best Validation Accuracy**: {best_fitness:.4f}
                - **Architecture Depth**: {best_architecture['num_layers']} layers
                - **Network Type**: {best_architecture.get('network_type', 'unknown')}
                """)
            except Exception as e:
                st.error(f"Error displaying summary: {str(e)}")
                return
            
            # Plot search progress
            try:
                st.subheader("Search Progress")
                # Check if history contains required fields for plotting
                if ('generations' not in history or not history['generations'] or 
                    'best_fitness' not in history or not history['best_fitness']):
                    st.warning("Cannot generate search progress plot - history data is incomplete")
                else:
                    fig = visualizer.plot_search_progress(history, metric="multiple")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating search progress plot: {str(e)}")
            
            # Plot architecture
            try:
                st.subheader("Architecture Visualization")
                
                # Create tabs for different visualization types
                vis_tabs = st.tabs(["Network Graph", "PyTorch Model"])
                
                with vis_tabs[0]:
                    fig = visualizer.visualize_architecture_networks([best_architecture], ["Best Architecture"])
                    st.pyplot(fig)
                    
                with vis_tabs[1]:
                    # Try to import the model builder
                    try:
                        from snas.architecture.model_builder import ModelBuilder
                        model_builder = ModelBuilder(device='cpu')
                        torch_fig = visualizer.visualize_torch_model(best_architecture, model_builder)
                        st.pyplot(torch_fig)
                        
                        # Add model summary if possible
                        st.subheader("Model Summary")
                        try:
                            if TORCH_AVAILABLE:
                                model = model_builder.build_model(best_architecture)
                                
                                # Count parameters
                                total_params = sum(p.numel() for p in model.parameters())
                                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                                
                                # Display summary
                                st.markdown(f"""
                                - **Total Parameters**: {total_params:,}
                                - **Trainable Parameters**: {trainable_params:,}
                                """)
                                
                                # Display model structure
                                st.code(str(model))
                            else:
                                st.warning("PyTorch is not available. Cannot display model summary.")
                            
                        except Exception as e:
                            st.warning(f"Could not generate model summary: {e}")
                        
                    except Exception as e:
                        st.warning(f"Could not generate PyTorch visualization: {e}")
                        st.info("Install torchviz with: pip install torchviz")
            except Exception as e:
                st.error(f"Error generating architecture visualization: {str(e)}")
            
            # Plot parameter importance
            try:
                st.subheader("Parameter Importance Analysis")
                # Check if history contains required fields for parameter importance
                if 'best_architecture' not in history or not history['best_architecture']:
                    st.warning("Cannot generate parameter importance analysis - history data is incomplete")
                else:
                    fig = visualizer.plot_parameter_importance(history, top_k=10)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating parameter importance analysis: {str(e)}")
            
            # Display architecture details
            try:
                st.subheader("Architecture Details")
                
                # Check if pandas is available
                if not PANDAS_AVAILABLE:
                    st.warning("Pandas is not available. Using simplified architecture view.")
                    
                    # Simple display without pandas DataFrames
                    network_type = best_architecture.get('network_type', 'unknown')
                    st.markdown(f"**Network Type**: {network_type}")
                    
                    # Display layer information based on network type
                    st.markdown("### Layer Information:")
                    
                    # CNN, ResNet, MobileNet, and other CNN-based architectures
                    if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
                        for i in range(best_architecture.get('num_layers', 0)):
                            layer_info = []
                            if 'filters' in best_architecture and i < len(best_architecture['filters']):
                                layer_info.append(f"Filters: {best_architecture['filters'][i]}")
                            if 'kernel_sizes' in best_architecture and i < len(best_architecture['kernel_sizes']):
                                layer_info.append(f"Kernel Size: {best_architecture['kernel_sizes'][i]}")
                            if 'activations' in best_architecture and i < len(best_architecture['activations']):
                                layer_info.append(f"Activation: {best_architecture['activations'][i]}")
                            if 'use_skip_connections' in best_architecture and i < len(best_architecture['use_skip_connections']):
                                layer_info.append(f"Skip Connection: {'Yes' if best_architecture['use_skip_connections'][i] else 'No'}")
                                
                            if layer_info:
                                st.markdown(f"**Layer {i+1}**: {', '.join(layer_info)}")
                                
                    # MLP and Enhanced MLP
                    elif network_type in ['mlp', 'enhanced_mlp']:
                        for i in range(best_architecture.get('num_layers', 0)):
                            layer_info = []
                            if 'hidden_units' in best_architecture and i < len(best_architecture['hidden_units']):
                                layer_info.append(f"Hidden Units: {best_architecture['hidden_units'][i]}")
                            if 'activations' in best_architecture and i < len(best_architecture['activations']):
                                layer_info.append(f"Activation: {best_architecture['activations'][i]}")
                                
                            if layer_info:
                                st.markdown(f"**Layer {i+1}**: {', '.join(layer_info)}")
                
                else:
                    try:
                        # Import pandas locally to avoid global import issues
                        import pandas as pd
                        
                        # Use pandas DataFrame for better display
                        arch_df = pd.DataFrame()
                        
                        # Layer information based on network type
                        network_type = best_architecture.get('network_type', 'unknown')
                        
                        # CNN, ResNet, MobileNet, and other CNN-based architectures
                        if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
                            if 'filters' in best_architecture:
                                arch_df['Filters'] = best_architecture['filters']
                            if 'kernel_sizes' in best_architecture:
                                arch_df['Kernel Size'] = best_architecture['kernel_sizes']
                            if 'activations' in best_architecture:
                                arch_df['Activation'] = best_architecture['activations']
                            if 'use_skip_connections' in best_architecture:
                                arch_df['Skip Connection'] = [
                                    "Yes" if x else "No" 
                                    for x in best_architecture['use_skip_connections']
                                ]
                        
                        # MLP and Enhanced MLP
                        elif network_type in ['mlp', 'enhanced_mlp']:
                            if 'hidden_units' in best_architecture:
                                arch_df['Hidden Units'] = best_architecture['hidden_units']
                            if 'activations' in best_architecture:
                                arch_df['Activation'] = best_architecture['activations']
                                
                        # Add layer names
                        if not arch_df.empty:
                            arch_df.index = [f"Layer {i+1}" for i in range(len(arch_df))]
                        
                            # Display the DataFrame
                            st.dataframe(arch_df)
                        else:
                            st.info("No layer-specific parameters found in this architecture")
                            
                    except ImportError:
                        # If pandas import fails here, fallback to the simpler view
                        st.warning("Pandas is not available. Using simplified architecture view.")
                        
                        # Simple display without pandas DataFrames
                        network_type = best_architecture.get('network_type', 'unknown')
                        st.markdown(f"**Network Type**: {network_type}")
                        
                        # Display layer information based on network type using the same code as above
                        st.markdown("### Layer Information:")
                        
                        # Just re-use the same code as the fallback display above
                        if network_type in ['cnn', 'resnet', 'mobilenet', 'densenet', 'shufflenetv2', 'efficientnet']:
                            for i in range(best_architecture.get('num_layers', 0)):
                                layer_info = []
                                if 'filters' in best_architecture and i < len(best_architecture['filters']):
                                    layer_info.append(f"Filters: {best_architecture['filters'][i]}")
                                if 'kernel_sizes' in best_architecture and i < len(best_architecture['kernel_sizes']):
                                    layer_info.append(f"Kernel Size: {best_architecture['kernel_sizes'][i]}")
                                if 'activations' in best_architecture and i < len(best_architecture['activations']):
                                    layer_info.append(f"Activation: {best_architecture['activations'][i]}")
                                if 'use_skip_connections' in best_architecture and i < len(best_architecture['use_skip_connections']):
                                    layer_info.append(f"Skip Connection: {'Yes' if best_architecture['use_skip_connections'][i] else 'No'}")
                                    
                                if layer_info:
                                    st.markdown(f"**Layer {i+1}**: {', '.join(layer_info)}")
                                    
                        # MLP and Enhanced MLP
                        elif network_type in ['mlp', 'enhanced_mlp']:
                            for i in range(best_architecture.get('num_layers', 0)):
                                layer_info = []
                                if 'hidden_units' in best_architecture and i < len(best_architecture['hidden_units']):
                                    layer_info.append(f"Hidden Units: {best_architecture['hidden_units'][i]}")
                                if 'activations' in best_architecture and i < len(best_architecture['activations']):
                                    layer_info.append(f"Activation: {best_architecture['activations'][i]}")
                                    
                                if layer_info:
                                    st.markdown(f"**Layer {i+1}**: {', '.join(layer_info)}")
                
                # Global parameters
                st.subheader("Global Parameters")
                global_params = {}
                
                # Common parameters for all network types
                for param in ['network_type', 'learning_rate', 'dropout_rate', 'optimizer', 'use_batch_norm']:
                    if param in best_architecture:
                        global_params[param] = best_architecture[param]
                
                # Network-specific parameters
                if network_type == 'mobilenet' and 'width_multiplier' in best_architecture:
                    global_params['width_multiplier'] = best_architecture['width_multiplier']
                elif network_type == 'densenet':
                    for param in ['growth_rate', 'compression_factor', 'bn_size']:
                        if param in best_architecture:
                            global_params[param] = best_architecture[param]
                elif network_type == 'shufflenetv2':
                    for param in ['width_multiplier']:
                        if param in best_architecture:
                            global_params[param] = best_architecture[param]
                elif network_type == 'efficientnet':
                    for param in ['width_factor', 'depth_factor', 'se_ratio']:
                        if param in best_architecture:
                            global_params[param] = best_architecture[param]
                elif network_type == 'enhanced_mlp':
                    for param in ['use_residual', 'use_layer_norm']:
                        if param in best_architecture:
                            global_params[param] = best_architecture[param]
                
                # Display the global parameters
                if global_params:
                    st.json(global_params)
                else:
                    st.info("No global parameters found in this architecture")
            except Exception as e:
                st.error(f"Error displaying architecture details: {str(e)}")
        else:
            st.info("Select a previous result from the sidebar to view details.")

    # Export tab
    with tab3:
        st.header("Export Architecture")
        
        if selected_result != "None":
            # Load selected results if not already loaded
            if 'best_architecture' not in locals():
                with st.spinner("Loading results..."):
                    history, best_architecture = load_search_results(selected_result)
            
            # Display options to export
            st.subheader("Export Options")
            
            # Export as PyTorch code
            if st.button("Generate PyTorch Code"):
                # Generate code for the best architecture
                code = f"""
# PyTorch Implementation of the Best Architecture from S-NAS
# Generated from: {selected_result}

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BestModel(nn.Module):
    def __init__(self, num_classes={best_architecture['num_classes']}):
        super(BestModel, self).__init__()
        
        # Model architecture
        self.num_layers = {best_architecture['num_layers']}
        self.input_shape = {best_architecture['input_shape']}
        self.network_type = "{best_architecture.get('network_type', 'cnn')}"
        
        # Layers
        """
                
                # Generate code based on network type
                network_type = best_architecture.get('network_type', 'cnn')
                
                if network_type == 'mlp':
                    # MLP architecture
                    hidden_units = best_architecture.get('hidden_units', [])
                    
                    code += f"""
        # Fully connected layers for MLP
        self.flatten = nn.Flatten()
        
        # Input layer
        input_size = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        
"""
                    for i in range(best_architecture['num_layers']):
                        if i < len(hidden_units):
                            input_dim = best_architecture['input_shape'][0] * best_architecture['input_shape'][1] * best_architecture['input_shape'][2] if i == 0 else hidden_units[i-1]
                            output_dim = hidden_units[i]
                            code += f"        self.fc{i+1} = nn.Linear({input_dim}, {output_dim})\n"
                    
                    # Output layer
                    if hidden_units:
                        code += f"        self.classifier = nn.Linear({hidden_units[-1]}, num_classes)\n"
                    else:
                        code += f"        self.classifier = nn.Linear(input_size, num_classes)\n"

                else:
                    # CNN, ResNet, or MobileNet architecture
                    in_channels = best_architecture['input_shape'][0]
                    filters = best_architecture.get('filters', [])
                    kernel_sizes = best_architecture.get('kernel_sizes', [])
                    
                    code += "        # Convolutional layers\n"
                    
                    for i in range(best_architecture['num_layers']):
                        if i < len(filters) and i < len(kernel_sizes):
                            out_channels = filters[i]
                            kernel_size = kernel_sizes[i]
                            code += f"""
        # Layer {i+1}
        self.conv{i+1} = nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, 
                                  kernel_size={kernel_size}, padding={kernel_size//2})
"""
                            if best_architecture.get('use_batch_norm', False):
                                code += f"        self.bn{i+1} = nn.BatchNorm2d({out_channels})\n"
                                
                            in_channels = out_channels
                    
                    # Add global average pooling and classifier
                    if filters:
                        code += f"""
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features={filters[-1]}, out_features=num_classes)
"""
                    else:
                        code += f"""
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features={in_channels}, out_features=num_classes)
"""
                
                # Add dropout
                code += f"""
        # Dropout
        self.dropout = nn.Dropout(p={best_architecture.get('dropout_rate', 0.0)})
"""
                
                # Forward method based on network type
                if network_type == 'mlp':
                    code += """
    def forward(self, x):
        # Flatten the input
        x = self.flatten(x)
        
"""
                    # Forward for each layer
                    for i in range(best_architecture['num_layers']):
                        activation = best_architecture.get('activations', ['relu'] * best_architecture['num_layers'])[i] if i < len(best_architecture.get('activations', [])) else 'relu'
                        code += f"""
        # Layer {i+1}
        x = self.fc{i+1}(x)
"""
                        # Activation
                        if activation == 'relu':
                            code += "        x = F.relu(x)\n"
                        elif activation == 'leaky_relu':
                            code += "        x = F.leaky_relu(x, 0.1)\n"
                        elif activation == 'elu':
                            code += "        x = F.elu(x)\n"
                        elif activation == 'selu':
                            code += "        x = F.selu(x)\n"
                        
                        # Dropout
                        code += "        x = self.dropout(x)\n"
                else:
                    # CNN, ResNet, or MobileNet forward
                    code += """
    def forward(self, x):
"""
                    # Forward for each convolutional layer
                    for i in range(best_architecture['num_layers']):
                        activation = best_architecture.get('activations', ['relu'] * best_architecture['num_layers'])[i] if i < len(best_architecture.get('activations', [])) else 'relu'
                        
                        code += f"""
        # Layer {i+1}
        x = self.conv{i+1}(x)
"""
                        # Batch normalization
                        if best_architecture.get('use_batch_norm', False):
                            code += f"        x = self.bn{i+1}(x)\n"
                            
                        # Activation
                        if activation == 'relu':
                            code += "        x = F.relu(x)\n"
                        elif activation == 'leaky_relu':
                            code += "        x = F.leaky_relu(x, 0.1)\n"
                        elif activation == 'elu':
                            code += "        x = F.elu(x)\n"
                        elif activation == 'selu':
                            code += "        x = F.selu(x)\n"
                    
                    # Add final global average pooling and classification
                    code += """
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        x = self.classifier(x)
"""
                
                # Complete the code with the return statement and model creation
                code += """
        return x

# Create model instance
model = BestModel()

# Optimizer configuration
optimizer = {}(
    model.parameters(),
    lr={},
    weight_decay=1e-4
)
""".format(
                    best_architecture.get('optimizer', 'Adam').capitalize(),
                    best_architecture.get('learning_rate', 0.001)
                )
                
                # Display code
                st.code(code, language="python")
                
                # Save code to file
                code_filename = os.path.join(OUTPUT_DIR, f"{selected_result}_model.py")
                with open(code_filename, 'w') as f:
                    f.write(code)
                    
                st.success(f"Code saved to: {code_filename}")
                
            # Export architecture as JSON
            if st.button("Export as JSON"):
                # Display architecture as JSON
                st.json(best_architecture)
                
                # Save to file if not already saved
                json_filename = os.path.join(RESULTS_DIR, f"{selected_result}_best.json")
                if not os.path.exists(json_filename):
                    with open(json_filename, 'w') as f:
                        json.dump(best_architecture, f, indent=2)
                        
                st.success(f"JSON saved to: {json_filename}")
        else:
            st.info("Select a previous result from the sidebar to export.")

    # Footer
    st.markdown("---")
    st.markdown("S-NAS: Simple Neural Architecture Search | Built with Streamlit and PyTorch")

if __name__ == "__main__":
    main()