import os
import json
import pickle
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from .exceptions import PersistenceError

logger = logging.getLogger(__name__)

class SearchStateManager:
    """Manages state for evolutionary search, enabling checkpointing and resuming."""
    
    def __init__(self, output_dir: str = "output", results_dir: str = "output/results"):
        """
        Initialize the state manager.
        
        Args:
            output_dir: Directory to store output files
            results_dir: Directory to store result files
        """
        self.output_dir = output_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Internal state
        self.current_checkpoint_path = None
    
    def save_checkpoint(self, 
                      dataset_name: str, 
                      search_state: Dict[str, Any], 
                      best_architecture: Dict[str, Any],
                      best_fitness: float,
                      history: Dict[str, List],
                      generation: int,
                      use_timestamp: bool = True) -> str:
        """
        Save current search state to a checkpoint file.
        
        Args:
            dataset_name: Name of the dataset being used
            search_state: Current search state including population and evaluated architectures
            best_architecture: Best architecture found so far
            best_fitness: Best fitness score found so far
            history: Search history
            generation: Current generation number
            use_timestamp: Whether to include timestamp in filename
            
        Returns:
            str: Path to the saved checkpoint file
        """
        try:
            # Create checkpoint filename
            if use_timestamp:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{dataset_name}_checkpoint_gen{generation}_{timestamp}"
            else:
                filename = f"{dataset_name}_checkpoint_gen{generation}"
            
            # Create full checkpoint object
            checkpoint = {
                'dataset_name': dataset_name,
                'search_state': search_state,
                'best_architecture': best_architecture,
                'best_fitness': best_fitness,
                'history': history,
                'generation': generation,
                'timestamp': time.time()
            }
            
            # Save checkpoint file
            checkpoint_path = os.path.join(self.results_dir, f"{filename}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Also save best architecture as JSON for easy inspection
            arch_path = os.path.join(self.results_dir, f"{filename}_best.json")
            with open(arch_path, 'w') as f:
                json.dump(best_architecture, f, indent=2)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            self.current_checkpoint_path = checkpoint_path
            return checkpoint_path
            
        except (IOError, OSError, pickle.PickleError) as e:
            error_msg = f"Failed to save checkpoint: {str(e)}"
            logger.error(error_msg)
            raise PersistenceError(error_msg, filepath=self.results_dir)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint from file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            dict: The loaded checkpoint
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            self.current_checkpoint_path = checkpoint_path
            return checkpoint
            
        except (IOError, OSError, pickle.PickleError) as e:
            error_msg = f"Failed to load checkpoint: {str(e)}"
            logger.error(error_msg)
            raise PersistenceError(error_msg, filepath=checkpoint_path)
    
    def save_final_results(self, dataset_name: str, 
                          search_history: Dict[str, List], 
                          best_architecture: Dict[str, Any], 
                          best_fitness: float) -> str:
        """
        Save final search results to disk.
        
        Args:
            dataset_name: Name of the dataset used
            search_history: Complete search history
            best_architecture: Best architecture found
            best_fitness: Best fitness score
            
        Returns:
            str: Base filename for the saved results
        """
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{dataset_name}_search_{timestamp}"
            
            # Save history
            with open(os.path.join(self.results_dir, f"{filename}_history.pkl"), 'wb') as f:
                pickle.dump(search_history, f)
            
            # Save best architecture as JSON
            with open(os.path.join(self.results_dir, f"{filename}_best.json"), 'w') as f:
                json.dump(best_architecture, f, indent=2)
            
            # Save a summary file
            summary = {
                'dataset': dataset_name,
                'best_fitness': best_fitness,
                'model_size': best_architecture.get('model_size', 'unknown'),
                'timestamp': timestamp,
                'num_layers': best_architecture.get('num_layers', 0),
                'network_type': best_architecture.get('network_type', 'unknown')
            }
            
            with open(os.path.join(self.results_dir, f"{filename}_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Final results saved: {filename}")
            return filename
            
        except (IOError, OSError, pickle.PickleError, json.JSONDecodeError) as e:
            error_msg = f"Failed to save final results: {str(e)}"
            logger.error(error_msg)
            raise PersistenceError(error_msg, filepath=self.results_dir)
    
    def get_available_checkpoints(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Get a list of available checkpoint files.
        
        Args:
            dataset_name: Optional filter by dataset name
            
        Returns:
            list: List of checkpoint filepaths
        """
        try:
            checkpoints = []
            
            # List all pickle files in the results directory
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.pkl') and 'checkpoint' in filename:
                    # Filter by dataset name if provided
                    if dataset_name and not filename.startswith(f"{dataset_name}_checkpoint"):
                        continue
                    
                    checkpoints.append(os.path.join(self.results_dir, filename))
            
            # Sort by generation and timestamp (newest first)
            checkpoints.sort(reverse=True)
            return checkpoints
            
        except OSError as e:
            error_msg = f"Failed to get available checkpoints: {str(e)}"
            logger.error(error_msg)
            raise PersistenceError(error_msg, filepath=self.results_dir)
    
    def load_latest_checkpoint(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            dict or None: The latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.get_available_checkpoints(dataset_name)
        
        if not checkpoints:
            logger.info(f"No checkpoints found for dataset {dataset_name}")
            return None
        
        latest_checkpoint = checkpoints[0]
        return self.load_checkpoint(latest_checkpoint)