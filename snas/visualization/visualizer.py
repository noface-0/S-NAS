"""
Visualizer for S-NAS

This module provides visualization utilities for the neural architecture search process.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional
import io
import base64

class SearchVisualizer:
    """Visualizes the neural architecture search process and results."""
    
    def __init__(self, architecture_space=None):
        """
        Initialize the visualizer.
        
        Args:
            architecture_space: The architecture space being explored
        """
        self.architecture_space = architecture_space
        
    def plot_search_progress(self, history, metric='best_fitness'):
        """
        Plot the progress of the search over generations.
        
        Args:
            history: Search history from EvolutionarySearch
            metric: Metric to plot ('best_fitness', 'avg_fitness', or 'population_diversity')
            
        Returns:
            fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        generations = history['generations']
        
        if metric == 'multiple':
            # Plot multiple metrics on the same graph
            ax.plot(generations, history['best_fitness'], 'b-', label='Best Fitness')
            ax.plot(generations, history['avg_fitness'], 'g-', label='Average Fitness')
            
            # Add diversity on a second axis
            ax2 = ax.twinx()
            ax2.plot(generations, history['population_diversity'], 'r--', label='Diversity')
            ax2.set_ylabel('Diversity', color='r')
            ax2.tick_params(axis='y', colors='r')
        else:
            # Plot a single metric
            ax.plot(generations, history[metric], 'b-')
            
        # Set labels and title
        ax.set_xlabel('Generation')
        if metric == 'best_fitness':
            ax.set_ylabel('Best Fitness')
            ax.set_title('Best Fitness Over Generations')
        elif metric == 'avg_fitness':
            ax.set_ylabel('Average Fitness')
            ax.set_title('Average Fitness Over Generations')
        elif metric == 'population_diversity':
            ax.set_ylabel('Population Diversity')
            ax.set_title('Population Diversity Over Generations')
        elif metric == 'multiple':
            ax.set_ylabel('Fitness')
            ax.set_title('Search Progress Over Generations')
            
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        if metric == 'multiple':
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        # Tight layout
        fig.tight_layout()
        return fig
    
    def plot_architecture_comparison(self, architectures, fitness_scores, labels=None):
        """
        Create a comparison visualization of different architectures.
        
        Args:
            architectures: List of architecture configurations
            fitness_scores: Fitness score for each architecture
            labels: Optional labels for each architecture
            
        Returns:
            fig: Matplotlib figure
        """
        # Check inputs
        if len(architectures) == 0:
            raise ValueError("No architectures provided for comparison")
            
        if len(architectures) != len(fitness_scores):
            raise ValueError("Number of architectures and fitness scores must match")
            
        # Create figure with subplots
        n = len(architectures)
        fig, axs = plt.subplots(n, 1, figsize=(12, n * 4))
        if n == 1:
            axs = [axs]
            
        # Create labels if not provided
        if labels is None:
            labels = [f"Architecture {i+1}" for i in range(n)]
            
        # Compare key features of each architecture
        for i, (arch, score, label) in enumerate(zip(architectures, fitness_scores, labels)):
            ax = axs[i]
            
            # Create a dictionary of architecture parameters to visualize
            arch_data = {}
            
            # Number of layers
            arch_data['num_layers'] = arch['num_layers']
            
            # Extract layer-specific data
            if 'filters' in arch:
                arch_data['avg_filters'] = np.mean(arch['filters'])
                arch_data['max_filters'] = max(arch['filters'])
                
            if 'kernel_sizes' in arch:
                arch_data['avg_kernel_size'] = np.mean(arch['kernel_sizes'])
                
            # Global parameters
            for param in ['learning_rate', 'dropout_rate']:
                if param in arch:
                    arch_data[param] = arch[param]
                    
            # Boolean parameters
            for param in ['use_batch_norm']:
                if param in arch:
                    arch_data[param] = 1 if arch[param] else 0
                    
            # Count skip connections if present
            if 'use_skip_connections' in arch:
                arch_data['skip_connections'] = sum(1 for x in arch['use_skip_connections'] if x)
                
            # Create bar plot of architecture parameters
            params = list(arch_data.keys())
            values = list(arch_data.values())
            
            # Normalize values for better visualization
            values_normalized = []
            for param, value in zip(params, values):
                if param == 'num_layers':
                    values_normalized.append(value / 10)  # Assuming max 10 layers
                elif param == 'avg_filters' or param == 'max_filters':
                    values_normalized.append(value / 256)  # Assuming max 256 filters
                elif param == 'learning_rate':
                    values_normalized.append(value * 100)  # Scale up small values
                elif param == 'avg_kernel_size':
                    values_normalized.append(value / 7)  # Assuming max kernel size 7
                else:
                    values_normalized.append(value)
                    
            # Plot bars
            ax.barh(params, values_normalized, alpha=0.7)
            
            # Add value labels
            for j, value in enumerate(values):
                if isinstance(value, float):
                    ax.text(values_normalized[j] + 0.05, j, f"{value:.4g}", va='center')
                else:
                    ax.text(values_normalized[j] + 0.05, j, f"{value}", va='center')
                    
            # Add fitness score and title
            ax.set_title(f"{label} - Fitness: {score:.4f}")
            
            # Customize appearance
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
        # Tight layout
        fig.tight_layout()
        return fig
    
    def visualize_architecture_networks(self, architectures, labels=None):
        """
        Visualize architectures as network graphs.
        
        Args:
            architectures: List of architecture configurations
            labels: Optional labels for each architecture
            
        Returns:
            fig: Matplotlib figure
        """
        # Check inputs
        if len(architectures) == 0:
            raise ValueError("No architectures provided for visualization")
            
        # Create labels if not provided
        if labels is None:
            labels = [f"Architecture {i+1}" for i in range(len(architectures))]
            
        # Create figure with subplots
        n = len(architectures)
        fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))
        if n == 1:
            axs = [axs]
            
        # Create network visualization for each architecture
        for i, (arch, label) in enumerate(zip(architectures, labels)):
            ax = axs[i]
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add input node
            G.add_node("Input", pos=(0, 0))
            
            # Add layer nodes
            num_layers = arch['num_layers']
            filters = arch['filters'] if 'filters' in arch else [64] * num_layers
            
            # Add layer nodes with sizes proportional to number of filters
            layer_nodes = []
            for j in range(num_layers):
                node_name = f"L{j+1}"
                layer_nodes.append(node_name)
                G.add_node(node_name, pos=(j+1, 0), filters=filters[j])
                
            # Add output node
            G.add_node("Output", pos=(num_layers+1, 0))
            
            # Add edges between consecutive layers
            G.add_edge("Input", layer_nodes[0])
            for j in range(num_layers-1):
                G.add_edge(layer_nodes[j], layer_nodes[j+1])
            G.add_edge(layer_nodes[-1], "Output")
            
            # Add skip connections if specified
            if 'use_skip_connections' in arch:
                for j in range(num_layers):
                    if arch['use_skip_connections'][j] and j > 0:  # Skip from earlier layers
                        # Find compatible earlier layers (same number of filters)
                        for k in range(j):
                            if filters[k] == filters[j]:
                                G.add_edge(layer_nodes[k], layer_nodes[j], style='dashed')
            
            # Get positions for all nodes
            pos = nx.get_node_attributes(G, 'pos')
            
            # Get filters for node sizes
            filters_dict = nx.get_node_attributes(G, 'filters')
            node_sizes = []
            for node in G.nodes:
                if node in filters_dict:
                    # Scale node size based on number of filters
                    node_sizes.append(filters_dict[node] * 2)
                elif node == "Input":
                    node_sizes.append(100)
                elif node == "Output":
                    node_sizes.append(100)
                else:
                    node_sizes.append(300)
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                                 node_color='skyblue', alpha=0.8)
            
            # Draw edges with different styles
            solid_edges = [(u, v) for u, v, e in G.edges(data=True) if 'style' not in e]
            dashed_edges = [(u, v) for u, v, e in G.edges(data=True) if e.get('style') == 'dashed']
            
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=solid_edges, width=2)
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dashed_edges, 
                                 width=1.5, style='dashed', edge_color='red', alpha=0.7)
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
            
            # Customize plot
            ax.set_title(label)
            ax.axis('off')
            
        # Tight layout
        fig.tight_layout()
        return fig
    
    def plot_training_curves(self, evaluation_results):
        """
        Plot training and validation curves from evaluation.
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            fig: Matplotlib figure
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract data
        epochs = range(1, len(evaluation_results['train_losses']) + 1)
        train_losses = evaluation_results['train_losses']
        val_losses = evaluation_results['val_losses']
        train_accs = evaluation_results['train_accs']
        val_accs = evaluation_results['val_accs']
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot accuracies
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        fig.tight_layout()
        return fig
    
    def plot_parameter_importance(self, history, top_k=10):
        """
        Analyze which architecture parameters are most important for performance.
        
        Args:
            history: Search history with best architectures
            top_k: Number of top architectures to analyze
            
        Returns:
            fig: Matplotlib figure
        """
        # Extract data from history
        best_architectures = history['best_architecture'][-top_k:]
        best_fitness = history['best_fitness'][-top_k:]
        
        # Create pandas DataFrame for parameter analysis
        data = []
        
        for arch, fitness in zip(best_architectures, best_fitness):
            arch_data = {'fitness': fitness}
            
            # Extract scalar parameters
            for param in ['num_layers', 'learning_rate', 'dropout_rate']:
                if param in arch:
                    arch_data[param] = arch[param]
            
            # Handle boolean parameters
            for param in ['use_batch_norm']:
                if param in arch:
                    arch_data[param] = 1 if arch[param] else 0
            
            # Extract layer-specific parameters
            if 'filters' in arch:
                arch_data['avg_filters'] = np.mean(arch['filters'])
                arch_data['max_filters'] = max(arch['filters'])
                
            if 'kernel_sizes' in arch:
                arch_data['avg_kernel_size'] = np.mean(arch['kernel_sizes'])
                
            # Count skip connections if present
            if 'use_skip_connections' in arch:
                arch_data['num_skip_connections'] = sum(1 for x in arch['use_skip_connections'] if x)
                
            data.append(arch_data)
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate correlations with fitness
        corr = df.corr()['fitness'].drop('fitness').sort_values(ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot correlations
        corr.plot(kind='bar', ax=ax)
        
        # Customize plot
        ax.set_title('Parameter Correlation with Performance')
        ax.set_xlabel('Architecture Parameter')
        ax.set_ylabel('Correlation with Fitness')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tight layout
        fig.tight_layout()
        return fig
    
    def fig_to_base64(self, fig):
        """
        Convert a matplotlib figure to base64 for embedding in HTML.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            str: Base64 encoded string
        """
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encode buffer as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str