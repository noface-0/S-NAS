import base64
import io
import os
import sys
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, Polygon
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

TORCH_AVAILABLE = False
TORCHVIZ_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    try:
        from torchviz import make_dot
        TORCHVIZ_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


class SearchVisualizer:
    """Visualizes the neural architecture search process and results."""

    def __init__(self, architecture_space=None):
        """
        Initialize the visualizer.

        Args:
            architecture_space: The architecture space being explored
        """
        self.architecture_space = architecture_space
        
        # Set default style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Default setting for metric direction
        self.higher_is_better = True  # Default: higher is better (accuracy)

    def plot_search_progress(self, history, metric='best_fitness'):
        """
        Plot the progress of the search over generations.

        Args:
            history: Search history from EvolutionarySearch
            metric: Metric to plot ('best_fitness', 'avg_fitness', 'population_diversity', or 'multiple')

        Returns:
            fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Extract data
        if 'generations' not in history:
            # For PNAS search, use complexity levels as generations
            if 'complexity_levels' in history:
                history['generations'] = history['complexity_levels']
            else:
                # If no generations or complexity levels, create a sequence
                if 'best_fitness' in history:
                    history['generations'] = list(range(1, len(history['best_fitness']) + 1))
                else:
                    # Fall back to a single point if no data available
                    history['generations'] = [1]
                    
        generations = history['generations']

        # Determine if we're dealing with a loss metric (lower is better)
        is_loss_metric = False
        # For accuracy metrics like 'val_acc', higher values are better
        # For loss metrics like 'val_loss', lower values are better
        
        # Set default values based on metric name
        if 'metric' in history:
            metric_name = history['metric']
            if metric_name.endswith('acc'):
                is_loss_metric = False
                self.higher_is_better = True
            elif metric_name.endswith('loss'):
                is_loss_metric = True
                self.higher_is_better = False
            else:
                # Try to infer from metric_type
                if 'metric_type' in history:
                    is_loss_metric = history['metric_type'] == 'loss'
                    self.higher_is_better = not is_loss_metric
                else:
                    # Default: treating as accuracy (higher is better)
                    is_loss_metric = False
                    self.higher_is_better = True
        elif 'metric_type' in history:
            is_loss_metric = history['metric_type'] == 'loss'
            self.higher_is_better = not is_loss_metric
        else:
            # Default to accuracy-based metric (higher is better)
            is_loss_metric = False
            self.higher_is_better = True

        # Get the metric name for better labels
        metric_name = history.get('metric', 'val_acc')
        
        if metric == 'multiple':
            # Plot multiple metrics on the same graph with enhanced styling
            best_label = f'Best {metric_name}'
            
            # Check if best_fitness exists
            if 'best_fitness' not in history:
                # Try to use beam_performances as an alternative for PNAS search
                if 'beam_performances' in history and history['beam_performances']:
                    # Use the last value from each level for plotting
                    history['best_fitness'] = [max(perf) if self.higher_is_better else min(perf) 
                                              for perf in history['beam_performances']]
                else:
                    # Create a dummy array if no fitness data exists
                    history['best_fitness'] = [0.5] * len(generations)
            
            best_line, = ax.plot(generations, history['best_fitness'], 'o-', color='#1f77b4',
                                 linewidth=2.5, markersize=6, label=best_label)
            
            # Check if avg_fitness exists (for evolutionary search) or create it (for PNAS)
            if 'avg_fitness' not in history:
                # For PNAS search, use average of beam performances
                if 'beam_performances' in history and history['beam_performances']:
                    history['avg_fitness'] = [sum(perf)/len(perf) if perf else 0 
                                             for perf in history['beam_performances']]
                else:
                    # Create a dummy array if no fitness data exists
                    history['avg_fitness'] = [0.3] * len(generations)
            
            avg_label = f'Average {metric_name}'
            avg_line, = ax.plot(generations, history['avg_fitness'], 's-', color='#2ca02c',
                                linewidth=2, markersize=5, label=avg_label)

            # Fill between best and average fitness to highlight improvement space
            ax.fill_between(generations,
                            history['best_fitness'],
                            history['avg_fitness'],
                            alpha=0.15, color='#1f77b4')

            # Add diversity - check if population_diversity exists
            ax2 = ax.twinx()
            
            if 'population_diversity' not in history:
                # For PNAS, we don't have diversity, so create a dummy or derive it
                if 'beam_architectures' in history:
                    # Create a simple diversity metric based on beam size variation
                    history['population_diversity'] = [len(set(str(arch))) / 1000 
                                                      for arch in history['beam_architectures']]
                else:
                    # Create a dummy array
                    history['population_diversity'] = [0.5] * len(generations)
            
            div_line, = ax2.plot(generations, history['population_diversity'], 'd-',
                                 color='#d62728', linewidth=1.5, markersize=5, label='Diversity')
            ax2.set_ylabel('Diversity', color='#d62728', fontweight='bold')
            ax2.tick_params(axis='y', colors='#d62728')
            ax2.spines['right'].set_color('#d62728')

            # Add complexity level tracking if available
            if 'complexity_level' in history and history['complexity_level']:
                ax3 = ax.twinx()
                # Offset the right spine of ax3 to prevent overlap with ax2
                ax3.spines['right'].set_position(('outward', 60))

                # Plot complexity level as step function
                complexity_levels = history['complexity_level']
                steps_x, steps_y = [], []

                for i, level in enumerate(complexity_levels):
                    if i == 0 or level != complexity_levels[i - 1]:
                        steps_x.append(generations[i])
                        steps_y.append(level)

                if steps_x and steps_y:
                    comp_line, = ax3.step(steps_x, steps_y, '-', color='#9467bd',
                                          linewidth=2, where='post', label='Complexity Level')
                    ax3.set_ylabel(
                        'Complexity Level',
                        color='#9467bd',
                        fontweight='bold')
                    ax3.tick_params(axis='y', colors='#9467bd')
                    ax3.set_yticks([1, 2, 3])
                    ax3.spines['right'].set_color('#9467bd')

                    # Get all handles and labels for legend
                    handles = [best_line, avg_line, div_line, comp_line]
                    labels = [
                        best_label,
                        avg_label,
                        'Diversity',
                        'Complexity Level']
                else:
                    handles = [best_line, avg_line, div_line]
                    labels = [best_label, avg_label, 'Diversity']
            else:
                handles = [best_line, avg_line, div_line]
                labels = [best_label, avg_label, 'Diversity']

            # Use the metric name we already retrieved
            if is_loss_metric:
                ax.set_ylabel(f'{metric_name} (lower is better)', fontweight='bold')
                ax.set_title(
                    f'Search Progress ({metric_name}) Over Generations',
                    fontsize=14,
                    fontweight='bold')
            else:
                ax.set_ylabel(f'{metric_name} (higher is better)', fontweight='bold')
                ax.set_title(
                    f'Search Progress ({metric_name}) Over Generations',
                    fontsize=14,
                    fontweight='bold')
        else:
            # Plot a single metric with enhanced styling
            if metric == 'population_diversity':
                ax.plot(generations, history[metric], 'd-', color='#d62728',
                        linewidth=2, markersize=6)

                # Smooth the diversity curve
                if len(generations) > 3:
                    try:
                        # Create smooth interpolated curve
                        from scipy.interpolate import make_interp_spline
                        x_smooth = np.linspace(
                            min(generations), max(generations), 100)
                        y_smooth = make_interp_spline(
                            generations, history[metric])(x_smooth)
                        ax.plot(
                            x_smooth,
                            y_smooth,
                            color='#d62728',
                            alpha=0.3,
                            linewidth=1.5)
                    except BaseException:
                        # Fall back if scipy is not available or errors
                        pass
            else:
                line = ax.plot(
                    generations,
                    history[metric],
                    'o-',
                    color='#1f77b4',
                    linewidth=2.5,
                    markersize=6)

            # Use the metric name we already retrieved at the top of the function
            
            if metric == 'best_fitness':
                if is_loss_metric:
                    ax.set_ylabel(
                        f'Best {metric_name} (lower is better)',
                        fontweight='bold')
                    ax.set_title(
                        f'Best {metric_name} Over Generations',
                        fontsize=14,
                        fontweight='bold')
                else:
                    ax.set_ylabel(f'Best {metric_name}', fontweight='bold')
                    ax.set_title(
                        f'Best {metric_name} Over Generations',
                        fontsize=14,
                        fontweight='bold')

            elif metric == 'avg_fitness':
                if is_loss_metric:
                    ax.set_ylabel(
                        f'Average {metric_name} (lower is better)',
                        fontweight='bold')
                    ax.set_title(
                        f'Average {metric_name} Over Generations',
                        fontsize=14,
                        fontweight='bold')
                else:
                    ax.set_ylabel(f'Average {metric_name}', fontweight='bold')
                    ax.set_title(
                        f'Average {metric_name} Over Generations',
                        fontsize=14,
                        fontweight='bold')

            elif metric == 'population_diversity':
                ax.set_ylabel('Population Diversity', fontweight='bold')
                ax.set_title(
                    'Population Diversity Over Generations',
                    fontsize=14,
                    fontweight='bold')

        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Generation', fontweight='bold')

        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Add legend with better positioning
        if metric == 'multiple':
            ax.legend(
                handles,
                labels,
                loc='best',
                frameon=True,
                framealpha=0.9)

        # Add generation markers for significant changes
        if 'best_fitness' in history:
            best_fitness = history['best_fitness']
            for i in range(1, len(best_fitness)):
                # Mark points where significant improvement happened
                if i < len(best_fitness) - 1:
                    improvement = (
                        best_fitness[i] - best_fitness[i - 1]) / max(abs(best_fitness[i - 1]), 1e-10)
                    threshold = 0.1 if not is_loss_metric else -0.1

                    if (not is_loss_metric and improvement > threshold) or (
                            is_loss_metric and improvement < threshold):
                        ax.axvline(
                            x=generations[i],
                            color='#ff7f0e',
                            linestyle='--',
                            alpha=0.5)

        # Tight layout
        fig.tight_layout()
        return fig

    def plot_architecture_comparison(
            self,
            architectures,
            fitness_scores,
            labels=None):
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
            raise ValueError(
                "Number of architectures and fitness scores must match")

        # Create figure with subplots
        n = len(architectures)
        fig, axs = plt.subplots(n, 1, figsize=(12, n * 4), dpi=100)
        if n == 1:
            axs = [axs]

        # Create labels if not provided
        if labels is None:
            labels = [f"Architecture {i+1}" for i in range(n)]

        # Define color palette
        palette = sns.color_palette("viridis", n_colors=10)

        # Compare key features of each architecture
        for i, (arch, score, label) in enumerate(
                zip(architectures, fitness_scores, labels)):
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

            if 'hidden_units' in arch:
                arch_data['avg_hidden_units'] = np.mean(arch['hidden_units'])
                arch_data['max_hidden_units'] = max(arch['hidden_units'])

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
                arch_data['skip_connections'] = sum(
                    1 for x in arch['use_skip_connections'] if x)

            # Create bar plot of architecture parameters
            params = list(arch_data.keys())
            values = list(arch_data.values())

            # Normalize values for better visualization
            values_normalized = []
            for param, value in zip(params, values):
                if param == 'num_layers':
                    values_normalized.append(
                        value / 16)  # Adjusted for max 16 layers
                elif param in ['avg_filters', 'max_filters']:
                    values_normalized.append(
                        value / 1024)  # Adjusted for max 1024 filters
                elif param == 'learning_rate':
                    values_normalized.append(
                        value * 1000)  # Scale up small values
                elif param == 'avg_kernel_size':
                    # Adjusted for max kernel size 9
                    values_normalized.append(value / 9)
                elif param in ['avg_hidden_units', 'max_hidden_units']:
                    values_normalized.append(
                        value / 4096)  # Adjusted for max hidden units
                else:
                    values_normalized.append(value)

            # Plot bars with color gradient
            bars = ax.barh(params, values_normalized, alpha=0.8, color=[
                           palette[j % len(palette)] for j in range(len(params))])

            # Add value labels
            for j, value in enumerate(values):
                if isinstance(value, float):
                    ax.text(
                        values_normalized[j] +
                        0.05,
                        j,
                        f"{value:.4g}",
                        va='center',
                        fontweight='bold')
                else:
                    ax.text(
                        values_normalized[j] +
                        0.05,
                        j,
                        f"{value}",
                        va='center',
                        fontweight='bold')

            # Add fitness score and title
            network_type = arch.get('network_type', 'Unknown')
            ax.set_title(
                f"{label} ({network_type}) - Fitness: {score:.4f}",
                fontsize=12,
                fontweight='bold')

            # Customize appearance
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Tight layout
        fig.tight_layout()
        return fig

    def visualize_architecture_networks(self, architectures, labels=None):
        """
        Visualize architectures as network graphs with improved styling.

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

        # Create figure with subplots arranged in a grid if there are many
        # architectures
        n = len(architectures)
        cols = min(3, n)  # Max 3 columns
        rows = (n + cols - 1) // cols  # Ceiling division

        fig = plt.figure(figsize=(cols * 5, rows * 5), dpi=100)

        # Create color maps
        cnn_cmap = plt.cm.Blues
        mlp_cmap = plt.cm.Greens
        resnet_cmap = plt.cm.Oranges
        mobile_cmap = plt.cm.Purples

        # Create subplots
        for i, (arch, label) in enumerate(zip(architectures, labels)):
            ax = fig.add_subplot(rows, cols, i + 1)

            # Determine the network type for appropriate styling
            network_type = arch.get('network_type', 'cnn')

            # Choose color based on network type
            if 'resnet' in network_type:
                node_cmap = resnet_cmap
                edge_color = 'orange'
                skip_color = 'red'
            elif 'mlp' in network_type:
                node_cmap = mlp_cmap
                edge_color = 'green'
                skip_color = 'darkgreen'
            elif 'mobile' in network_type:
                node_cmap = mobile_cmap
                edge_color = 'purple'
                skip_color = 'darkviolet'
            else:
                node_cmap = cnn_cmap
                edge_color = 'blue'
                skip_color = 'darkblue'

            # Create a directed graph
            G = nx.DiGraph()

            # Add input node
            G.add_node("Input", pos=(0, 0), type='input')

            # Add layer nodes
            num_layers = arch['num_layers']

            # Determine node sizes based on architecture type
            if 'filters' in arch:
                sizes = arch['filters']
                size_label = 'filters'
            elif 'hidden_units' in arch:
                sizes = arch['hidden_units']
                size_label = 'units'
            else:
                sizes = [100] * num_layers
                size_label = None

            # Get activations if available
            if 'activations' in arch:
                activations = arch['activations']
            else:
                activations = ['relu'] * num_layers

            # Add layer nodes with sizes proportional to filters/units
            layer_nodes = []
            max_size = max(sizes) if sizes else 100

            for j in range(num_layers):
                node_name = f"L{j+1}"
                layer_nodes.append(node_name)

                # Calculate node size proportional to number of filters/units
                if j < len(sizes):
                    size = sizes[j]
                    size_normalized = max(20, 100 * (size / max_size))
                else:
                    size = 100
                    size_normalized = 50

                # Add activation info
                if j < len(activations):
                    activation = activations[j]
                else:
                    activation = 'relu'

                G.add_node(
                    node_name,
                    pos=(
                        j + 1,
                        0),
                    size=size,
                    size_normalized=size_normalized,
                    activation=activation,
                    type='layer')

            # Add output node
            G.add_node("Output", pos=(num_layers + 1, 0), type='output')

            # Add edges between consecutive layers
            G.add_edge("Input", layer_nodes[0], type='forward')
            for j in range(num_layers - 1):
                G.add_edge(layer_nodes[j], layer_nodes[j + 1], type='forward')
            G.add_edge(layer_nodes[-1], "Output", type='forward')

            # Add skip connections if specified
            if 'use_skip_connections' in arch:
                for j in range(num_layers):
                    if j < len(
                            arch['use_skip_connections']) and arch['use_skip_connections'][j] and j > 0:
                        # Add skip connection from earlier layers with
                        # compatible dimensions
                        for k in range(j):
                            if k < len(sizes) and j < len(
                                    sizes) and sizes[k] == sizes[j]:
                                G.add_edge(
                                    layer_nodes[k], layer_nodes[j], type='skip')

            # Get positions for all nodes
            pos = nx.get_node_attributes(G, 'pos')

            # Get node types
            node_types = nx.get_node_attributes(G, 'type')

            # Draw nodes with different styles based on type
            input_nodes = [n for n, t in node_types.items() if t == 'input']
            layer_nodes_all = [
                n for n, t in node_types.items() if t == 'layer']
            output_nodes = [n for n, t in node_types.items() if t == 'output']

            # Get node sizes
            node_sizes = []
            node_colors = []

            for node in G.nodes():
                if node in layer_nodes_all:
                    # Scale node size and get appropriate color
                    size_norm = G.nodes[node]['size_normalized']
                    node_sizes.append(size_norm)

                    # Color based on position in network (darker as layers go
                    # deeper)
                    layer_idx = int(node[1:]) - 1  # Extract layer number
                    # Scale from 0.3 to 1.0
                    color_val = 0.3 + 0.7 * (layer_idx / max(num_layers, 1))
                    node_colors.append(node_cmap(color_val))
                elif node == "Input":
                    node_sizes.append(80)
                    node_colors.append('lightgray')
                elif node == "Output":
                    node_sizes.append(80)
                    node_colors.append('lightgray')

            # Draw the nodes - use fixed size and color to avoid errors
            if layer_nodes_all:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=ax,
                    nodelist=layer_nodes_all,
                    node_size=300,  # Fixed size instead of variable
                    node_color='lightblue',  # Fixed color instead of variable
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.9)

            # Draw input and output nodes with different shapes
            if input_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=ax,
                    nodelist=input_nodes,
                    node_size=80,
                    node_shape='s',
                    node_color='lightgray',
                    edgecolors='black',
                    linewidths=1)
            if output_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=ax,
                    nodelist=output_nodes,
                    node_size=80,
                    node_shape='s',
                    node_color='lightgray',
                    edgecolors='black',
                    linewidths=1)

            # Draw edges with different styles
            forward_edges = [
                (u, v) for u, v, d in G.edges(
                    data=True) if d.get('type') == 'forward']
            skip_edges = [
                (u, v) for u, v, d in G.edges(
                    data=True) if d.get('type') == 'skip']

            if forward_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax,
                    edgelist=forward_edges,
                    width=1.5,
                    edge_color=edge_color,
                    arrows=True,
                    arrowsize=15,
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.1')

            if skip_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax,
                    edgelist=skip_edges,
                    width=1.5,
                    edge_color=skip_color,
                    style='dashed',
                    arrows=True,
                    arrowsize=15,
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.3')

            # Draw node labels with custom formatting
            node_labels = {}
            for node in G.nodes():
                if node in layer_nodes_all:
                    size = G.nodes[node]['size']
                    activation = G.nodes[node]['activation']
                    node_labels[node] = f"{node}\n{size} {size_label}\n{activation}"
                else:
                    node_labels[node] = node

            nx.draw_networkx_labels(
                G,
                pos,
                ax=ax,
                labels=node_labels,
                font_size=8,
                font_weight='bold')

            # Add legend for skip connections if present
            if skip_edges:
                import matplotlib.lines as mlines
                forward_line = mlines.Line2D(
                    [],
                    [],
                    color=edge_color,
                    linestyle='-',
                    linewidth=1.5,
                    label='Forward Connection')
                skip_line = mlines.Line2D(
                    [],
                    [],
                    color=skip_color,
                    linestyle='--',
                    linewidth=1.5,
                    label='Skip Connection')
                ax.legend(
                    handles=[
                        forward_line, skip_line], loc='upper center', bbox_to_anchor=(
                        0.5, -0.05), ncol=2)

            # Customize plot
            ax.set_title(
                f"{label} ({network_type})\n{num_layers} layers",
                fontsize=12,
                fontweight='bold')
            ax.axis('off')

            # Add background gradient based on architecture type
            background_color = node_cmap(0.1)
            ax.set_facecolor(background_color)

        # Tight layout
        fig.tight_layout()
        return fig

    def plot_training_curves(self, evaluation_results):
        """
        Plot training and validation curves from evaluation with enhanced styling.

        Args:
            evaluation_results: Results from model evaluation

        Returns:
            fig: Matplotlib figure
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        # Extract data
        epochs = range(1, len(evaluation_results['train_losses']) + 1)
        train_losses = evaluation_results['train_losses']
        val_losses = evaluation_results['val_losses']
        train_accs = evaluation_results['train_accs']
        val_accs = evaluation_results['val_accs']

        # Plot losses with enhanced styling
        train_line, = ax1.plot(epochs, train_losses, 'o-', color='#1f77b4',
                               linewidth=2, markersize=5, label='Training Loss')
        val_line, = ax1.plot(epochs, val_losses, 's-', color='#d62728',
                             linewidth=2, markersize=5, label='Validation Loss')

        # Fill between training and validation loss
        ax1.fill_between(epochs, train_losses, val_losses, alpha=0.2,
                         color='gray' if val_losses[-1] < train_losses[-1] else '#d62728')

        # Mark the best validation loss point
        best_val_loss_idx = np.argmin(val_losses)
        ax1.scatter([epochs[best_val_loss_idx]], [val_losses[best_val_loss_idx]],
                    s=[100], color='#d62728', edgecolors='black', zorder=5)
        ax1.annotate(
            f'Best: {val_losses[best_val_loss_idx]:.4f}',
            xy=(
                epochs[best_val_loss_idx],
                val_losses[best_val_loss_idx]),
            xytext=(
                epochs[best_val_loss_idx] +
                0.5,
                val_losses[best_val_loss_idx] -
                0.05),
            fontsize=10,
            fontweight='bold')

        # Customize loss plot
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title(
            'Training and Validation Loss',
            fontsize=14,
            fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot accuracies with enhanced styling
        train_acc_line, = ax2.plot(epochs, train_accs, 'o-', color='#1f77b4',
                                   linewidth=2, markersize=5, label='Training Accuracy')
        val_acc_line, = ax2.plot(epochs, val_accs, 's-', color='#2ca02c',
                                 linewidth=2, markersize=5, label='Validation Accuracy')

        # Fill between training and validation accuracy
        ax2.fill_between(epochs, train_accs, val_accs, alpha=0.2,
                         color='gray' if val_accs[-1] > train_accs[-1] else '#2ca02c')

        # Mark the best validation accuracy point
        best_val_acc_idx = np.argmax(val_accs)
        ax2.scatter([epochs[best_val_acc_idx]], [val_accs[best_val_acc_idx]],
                    s=[100], color='#2ca02c', edgecolors='black', zorder=5)
        ax2.annotate(f'Best: {val_accs[best_val_acc_idx]:.4f}',
                     xy=(epochs[best_val_acc_idx], val_accs[best_val_acc_idx]),
                     xytext=(epochs[best_val_acc_idx] + 0.5, val_accs[best_val_acc_idx] - 0.05),
                     fontsize=10, fontweight='bold')

        # Customize accuracy plot
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title(
            'Training and Validation Accuracy',
            fontsize=14,
            fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Set y-axis range for accuracy plot (0-1)
        ax2.set_ylim(0, 1.05)

        # Highlight early stopping point if available
        if 'early_stopping_epoch' in evaluation_results:
            es_epoch = evaluation_results['early_stopping_epoch']
            if es_epoch is not None and es_epoch < len(epochs):
                ax1.axvline(
                    x=es_epoch,
                    color='black',
                    linestyle='--',
                    alpha=0.7)
                ax2.axvline(
                    x=es_epoch,
                    color='black',
                    linestyle='--',
                    alpha=0.7)

                # Add annotation
                ax1.text(
                    es_epoch + 0.1,
                    max(train_losses) * 0.8,
                    'Early Stopping',
                    rotation=90,
                    fontsize=10,
                    fontweight='bold')

        # Add general information if available
        if 'train_time' in evaluation_results:
            train_time = evaluation_results['train_time']
            train_info = f"Training time: {train_time:.2f}s"
            fig.text(
                0.5,
                0.01,
                train_info,
                ha='center',
                fontsize=10,
                fontweight='bold')

        # Tight layout
        fig.tight_layout()
        return fig

    def plot_parameter_importance(self, history, top_k=10):
        """
        Analyze which architecture parameters are most important for performance
        with enhanced visualization.

        Args:
            history: Search history with best architectures
            top_k: Number of top architectures to analyze

        Returns:
            fig: Matplotlib figure with correlation and parameter distribution
        """
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(14, 8), dpi=100)
        
        try:
            # Extract data from history
            best_architectures = history['best_architecture'][-top_k:]
            best_fitness = history['best_fitness'][-top_k:]
            
            # Determine metric type (accuracy vs loss)
            is_loss_metric = False
            if 'metric_type' in history:
                is_loss_metric = history['metric_type'] == 'loss'
            elif 'metric' in history and history['metric'].endswith('loss'):
                is_loss_metric = True
            
            # Add a title that reflects the metric type
            metric_name = history.get('metric', 'Performance')
            if is_loss_metric:
                plt.suptitle(f"Parameter Importance Analysis (Loss metric: {metric_name})", 
                         fontsize=16, fontweight='bold')
            else:
                plt.suptitle(f"Parameter Importance Analysis (Accuracy metric: {metric_name})", 
                         fontsize=16, fontweight='bold')
    
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
    
                if 'hidden_units' in arch:
                    arch_data['avg_hidden_units'] = np.mean(arch['hidden_units'])
                    arch_data['max_hidden_units'] = max(arch['hidden_units'])
    
                # Count skip connections if present
                if 'use_skip_connections' in arch:
                    arch_data['num_skip_connections'] = sum(
                        1 for x in arch['use_skip_connections'] if x)
    
                # Extract activation function distribution
                if 'activations' in arch:
                    activation_types = set(arch['activations'])
                    for act_type in activation_types:
                        count = arch['activations'].count(act_type)
                        arch_data[f'activation_{act_type}'] = count / \
                            len(arch['activations'])
    
                data.append(arch_data)
    
            # Create DataFrame
            df = pd.DataFrame(data)
    
            # Calculate correlations with fitness
            corr = df.corr()['fitness'].drop('fitness')
    
            # Sort correlations by absolute value
            corr_abs = corr.abs().sort_values(ascending=False)
            corr = corr[corr_abs.index]
    
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    
            # 1. Correlation plot
            ax1 = fig.add_subplot(gs[0])
    
            # Plot correlations with custom coloring
            bars = ax1.barh(corr.index, corr.values, color=[
                '#1f77b4' if v > 0 else '#d62728' for v in corr.values
            ], alpha=0.7)
    
            # Add value labels with check for finite values
            for i, bar in enumerate(bars):
                if i < len(corr.values):
                    value = corr.values[i]
                    # Check if value is finite before positioning text
                    if np.isfinite(value):
                        text_x = value + (0.02 if value >= 0 else -0.02)
                        if np.isfinite(text_x) and np.isfinite(i):
                            ax1.text(text_x, i, f"{value:.3f}",
                                ha='left' if value >= 0 else 'right',
                                va='center', fontweight='bold', fontsize=9)
    
            # Customize plot based on metric type (accuracy vs loss)
            ax1.set_title(
                'Parameter Correlation with Performance',
                fontsize=14,
                fontweight='bold')
            ax1.set_xlabel('Correlation Coefficient', fontweight='bold')
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
            # Add correlation interpretation with position check
            if is_loss_metric:
                corr_text = "Strong negative correlation: Parameter decreases loss (better)\n"
                corr_text += "Strong positive correlation: Parameter increases loss (worse)"
            else:
                corr_text = "Strong positive correlation: Parameter increases accuracy (better)\n"
                corr_text += "Strong negative correlation: Parameter decreases accuracy (worse)"
                
            # Use safe transform coordinates
            ax1.text(0.02, -0.08, corr_text, transform=ax1.transAxes, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
            # 2. Parameter distribution plot for top 3 most correlated parameters
            ax2 = fig.add_subplot(gs[1])
    
            # Get top 3 most correlated parameters (by absolute value)
            top_params = corr_abs.index[:min(3, len(corr_abs))]
    
            if len(top_params) > 0:
                # Create box plots for each parameter
                box_data = [df[param].values for param in top_params]
                box_plot = ax2.boxplot(box_data, vert=False, patch_artist=True, widths=0.6, labels=[
                                    param if len(param) < 15 else param[:12] + '...' for param in top_params])
    
                # Customize box appearance
                for i, box in enumerate(box_plot['boxes']):
                    box.set(facecolor=plt.cm.viridis(i / max(1, len(box_plot['boxes']) - 1)), alpha=0.7)
    
                # Customize plot
                ax2.set_title(
                    'Distribution of Top Parameters',
                    fontsize=14,
                    fontweight='bold')
                ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
                # Add scatter plot points for individual values with position checking
                for i, param in enumerate(top_params):
                    # Get the y position of the boxplot
                    y_pos = i + 1
    
                    # Add scatter points with slight jitter
                    jitter = np.random.normal(0, 0.05, size=len(df[param]))
                    
                    # Check for finite values
                    valid_indices = np.isfinite(df[param])
                    if np.any(valid_indices):
                        scatter_size = np.full(np.sum(valid_indices), 50)
                        ax2.scatter(
                            df[param][valid_indices],
                            y_pos + jitter[valid_indices],
                            alpha=0.6,
                            c=df['fitness'][valid_indices],
                            cmap='viridis',
                            edgecolor='k',
                            s=scatter_size)
    
                # Add colorbar for fitness values
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), cax=cax)
                cbar.set_label('Fitness', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, "Not enough data for distribution plot", 
                         ha='center', va='center', fontsize=10, transform=ax2.transAxes)
                
        except Exception as e:
            # If any error occurs, display an error message
            plt.clf()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error generating parameter importance plot: {str(e)}",
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')

        # Tight layout
        fig.tight_layout()
        return fig

    def visualize_model_comparison(
            self,
            architectures,
            evaluations,
            labels=None):
        """
        Create an advanced comparison of different model architectures with their
        performance metrics.

        Args:
            architectures: List of architecture configurations
            evaluations: List of evaluation results
            labels: Optional labels for each architecture

        Returns:
            fig: Matplotlib figure
        """
        # Check inputs
        if len(architectures) == 0:
            raise ValueError("No architectures provided for comparison")

        if len(architectures) != len(evaluations):
            raise ValueError(
                "Number of architectures and evaluations must match")

        # Create labels if not provided
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(architectures))]

        # Create figure with grid layout
        n = len(architectures)
        fig = plt.figure(figsize=(16, n * 4), dpi=100)

        # Create grid specification
        gs = gridspec.GridSpec(n, 3, figure=fig, width_ratios=[1.5, 1, 1])

        # Define colors for different network types
        network_colors = {
            'cnn': '#1f77b4',      # Blue
            'mlp': '#2ca02c',      # Green
            'enhanced_mlp': '#98df8a',  # Light green
            'resnet': '#ff7f0e',   # Orange
            'mobilenet': '#9467bd',  # Purple
            'densenet': '#d62728',  # Red
            'shufflenetv2': '#e377c2',  # Pink
            'efficientnet': '#7f7f7f'  # Gray
        }

        # Create model cards for each architecture
        for i, (arch, eval_result, label) in enumerate(
                zip(architectures, evaluations, labels)):
            # Get network type and color
            network_type = arch.get('network_type', 'Unknown')
            color = network_colors.get(network_type, '#1f77b4')

            # 1. Architecture visualization (left panel)
            ax1 = fig.add_subplot(gs[i, 0])

            # Create network diagram
            self._create_network_diagram(ax1, arch, color)

            # 2. Performance metrics (middle panel)
            ax2 = fig.add_subplot(gs[i, 1])

            # Extract metrics from evaluation
            metrics = {
                'Test Accuracy': eval_result.get('test_acc', 0),
                'Val Accuracy': eval_result.get('best_val_acc', 0),
                'Train Accuracy': eval_result.get('train_accs', [0])[-1],
                'Parameters': eval_result.get('num_parameters', 0),
                'Training Time': eval_result.get('train_time', 0)
            }

            # Format metrics for display
            formatted_metrics = {}
            for k, v in metrics.items():
                if k == 'Parameters':
                    formatted_metrics[k] = f"{v:,}"
                elif k == 'Training Time':
                    formatted_metrics[k] = f"{v:.2f}s"
                else:
                    formatted_metrics[k] = f"{v:.4f}"

            # Create metrics table
            self._create_metrics_table(ax2, formatted_metrics, color)

            # 3. Architecture properties (right panel)
            ax3 = fig.add_subplot(gs[i, 2])

            # Extract key properties
            properties = {
                'Network Type': network_type, 'Layers': arch.get(
                    'num_layers', 0), 'Batch Norm': 'Yes' if arch.get(
                    'use_batch_norm', False) else 'No', 'Dropout': arch.get(
                    'dropout_rate', 0), 'Learning Rate': arch.get(
                    'learning_rate', 0)}

            # Add network-specific properties
            if 'filters' in arch:
                properties['Filters'] = f"{min(arch['filters'])} - {max(arch['filters'])}"
            if 'kernel_sizes' in arch:
                properties['Kernel Sizes'] = f"{min(arch['kernel_sizes'])} - {max(arch['kernel_sizes'])}"
            if 'hidden_units' in arch:
                properties['Hidden Units'] = f"{min(arch['hidden_units'])} - {max(arch['hidden_units'])}"
            if 'use_skip_connections' in arch:
                skip_count = sum(1 for x in arch['use_skip_connections'] if x)
                properties['Skip Connections'] = f"{skip_count}/{len(arch['use_skip_connections'])}"

            # Create properties table
            self._create_properties_table(ax3, properties, color)

            # Add model title at the top spanning all columns
            row_height = 1 / n
            title_y = 1 - i * row_height - row_height / 2
            fig.text(0.5, title_y + 0.03, f"{label}", ha='center', va='bottom',
                     fontsize=14, fontweight='bold', color='black')

        # Adjust layout
        fig.tight_layout()
        return fig

    def _create_network_diagram(self, ax, architecture, color='#1f77b4'):
        """Helper method to create a simplified network diagram."""
        try:
            # Get architecture details
            num_layers = architecture.get('num_layers', 1)
            network_type = architecture.get('network_type', 'Unknown')
    
            # Determine node sizes based on architecture type
            if 'filters' in architecture:
                sizes = architecture['filters']
                size_label = 'filters'
            elif 'hidden_units' in architecture:
                sizes = architecture['hidden_units']
                size_label = 'units'
            else:
                sizes = [64] * num_layers
                size_label = 'nodes'
    
            # Create boxes for visualization
            max_size = max(sizes) if sizes else 64
            layer_height = 0.7  # maximum height for the largest layer
    
            # Draw input layer
            input_x, input_y = 0.1, 0.5
            input_width, input_height = 0.1, 0.2
            ax.add_patch(
                Rectangle(
                    (input_x,
                     input_y -
                     input_height /
                     2),
                    input_width,
                    input_height,
                    facecolor='lightgray',
                    edgecolor='black',
                    alpha=0.7))
            
            # Add text with safe coordinate check
            if np.isfinite(input_x + input_width / 2) and np.isfinite(input_y):
                ax.text(input_x + input_width / 2, input_y, "Input",
                    ha='center', va='center', fontsize=8, fontweight='bold')
    
            # Draw hidden layers
            for i in range(num_layers):
                if i < len(sizes):
                    size = sizes[i]
                    height = layer_height * (size / max_size)
                    width = 0.1
                else:
                    size = 64
                    height = layer_height * 0.5
                    width = 0.1
    
                x = 0.3 + i * 0.5 / max(1, num_layers)
                y = 0.5
    
                # Draw the layer
                ax.add_patch(
                    Rectangle(
                        (x,
                         y - height / 2),
                        width,
                        height,
                        facecolor=color,
                        edgecolor='black',
                        alpha=0.7))
    
                # Layer label - add with safe coordinate check
                if np.isfinite(x + width / 2) and np.isfinite(y):
                    ax.text(
                        x + width / 2,
                        y,
                        f"L{i+1}",
                        ha='center',
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        color='white')
    
                # Size label - add with safe coordinate check
                if i < len(sizes) and np.isfinite(x + width / 2) and np.isfinite(y - height / 2 - 0.05):
                    ax.text(
                        x + width / 2,
                        y - height / 2 - 0.05,
                        f"{sizes[i]} {size_label}",
                        ha='center',
                        va='top',
                        fontsize=7)
    
                # Draw connection from previous layer - check coordinates are valid
                if i == 0:
                    # Connect from input
                    if np.isfinite(input_x + input_width) and np.isfinite(x) and np.isfinite(input_y) and np.isfinite(y):
                        ax.plot([input_x + input_width, x], [input_y, y],
                                color='black', linestyle='-', linewidth=1)
                else:
                    # Connect from previous hidden layer
                    prev_x = 0.3 + (i - 1) * 0.5 / max(1, num_layers)
                    if np.isfinite(prev_x + width) and np.isfinite(x) and np.isfinite(y):
                        ax.plot([prev_x + width, x], [y, y],
                                color='black', linestyle='-', linewidth=1)
    
            # Draw output layer
            output_x = 0.3 + num_layers * 0.5 / max(1, num_layers) + 0.1
            output_y = 0.5
            output_width, output_height = 0.1, 0.2
            
            if np.isfinite(output_x) and np.isfinite(output_y - output_height / 2):
                ax.add_patch(
                    Rectangle(
                        (output_x,
                         output_y -
                         output_height /
                         2),
                        output_width,
                        output_height,
                        facecolor='lightgray',
                        edgecolor='black',
                        alpha=0.7))
                        
                # Add output text with coordinates check
                if np.isfinite(output_x + output_width / 2) and np.isfinite(output_y):
                    ax.text(output_x + output_width / 2, output_y, "Output",
                            ha='center', va='center', fontsize=8, fontweight='bold')
    
            # Connect last hidden layer to output
            last_hidden_x = 0.3 + (num_layers - 1) * 0.5 / max(1, num_layers)
            if np.isfinite(last_hidden_x + width) and np.isfinite(output_x) and np.isfinite(y) and np.isfinite(output_y):
                ax.plot([last_hidden_x + width, output_x], [y, output_y],
                        color='black', linestyle='-', linewidth=1)
    
            # Add skip connections if specified
            if 'use_skip_connections' in architecture:
                for j in range(num_layers):
                    if j < len(architecture['use_skip_connections']
                               ) and architecture['use_skip_connections'][j] and j > 0:
                        # Add skip connection from earlier layers
                        for k in range(j - 1):
                            src_x = 0.3 + k * 0.5 / max(1, num_layers) + width
                            dest_x = 0.3 + j * 0.5 / max(1, num_layers)
                            
                            # Check coordinates before drawing
                            if np.isfinite(src_x) and np.isfinite(dest_x) and np.isfinite(y + 0.1):
                                # Draw arced skip connection
                                ax.annotate("",
                                            xy=(dest_x, y + 0.1),
                                            xytext=(src_x, y + 0.1),
                                            arrowprops=dict(arrowstyle="->",
                                                            color="red",
                                                            linestyle="dashed",
                                                            linewidth=1.5,
                                                            connectionstyle="arc3,rad=0.3"))
    
            # Add title and network type
            ax.set_title(
                f"{network_type.upper()} Architecture",
                fontsize=10,
                fontweight='bold')
    
            # Remove axes
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
        except Exception as e:
            # If any error occurs, display a simple error message
            ax.clear()
            ax.text(0.5, 0.5, f"Error rendering network diagram: {str(e)}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _create_metrics_table(self, ax, metrics, color='#1f77b4'):
        """Helper method to create a formatted metrics table."""
        try:
            # Turn off axes
            ax.axis('off')
    
            # Title
            ax.text(0.5, 0.95, "Performance Metrics", ha='center', va='top',
                    fontsize=12, fontweight='bold')
    
            # Create background for table
            ax.add_patch(Rectangle((0.1, 0.15), 0.8, 0.7,
                                facecolor='#f8f9fa', edgecolor=color,
                                linewidth=2, alpha=0.7, zorder=1))
    
            # Add metrics as text
            y_pos = 0.85
            for key, value in metrics.items():
                # Check coordinates are finite
                if np.isfinite(y_pos):
                    ax.text(0.15, y_pos, key, ha='left', va='center',
                            fontsize=10, fontweight='bold', zorder=2)
                    ax.text(0.85, y_pos, value, ha='right', va='center',
                            fontsize=10, zorder=2)
        
                    # Add separator line
                    ax.plot([0.1, 0.9], [y_pos - 0.07, y_pos - 0.07],
                            color='gray', linestyle='--', alpha=0.5, zorder=2)
        
                y_pos -= 0.15
                
                # Safety check - stop if we've gone too far
                if y_pos < 0.1:
                    break
        except Exception as e:
            # If any error occurs, clear and display a message
            ax.clear()
            ax.text(0.5, 0.5, f"Error creating metrics table: {str(e)}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def _create_properties_table(self, ax, properties, color='#1f77b4'):
        """Helper method to create a formatted properties table."""
        try:
            # Turn off axes
            ax.axis('off')
    
            # Title
            ax.text(0.5, 0.95, "Architecture Properties", ha='center', va='top',
                    fontsize=12, fontweight='bold')
    
            # Create background for table
            ax.add_patch(Rectangle((0.1, 0.15), 0.8, 0.7,
                                facecolor='#f8f9fa', edgecolor=color,
                                linewidth=2, alpha=0.7, zorder=1))
    
            # Add properties as text
            y_pos = 0.85
            for key, value in properties.items():
                # Check coordinates are finite
                if np.isfinite(y_pos):
                    ax.text(0.15, y_pos, key, ha='left', va='center',
                            fontsize=10, fontweight='bold', zorder=2)
                    ax.text(0.85, y_pos, str(value), ha='right', va='center',
                            fontsize=10, zorder=2)
    
                    # Add separator line
                    ax.plot([0.1, 0.9], [y_pos - 0.07, y_pos - 0.07],
                            color='gray', linestyle='--', alpha=0.5, zorder=2)
    
                y_pos -= 0.15
                
                # Safety check - stop if we've gone too far
                if y_pos < 0.1:
                    break
        except Exception as e:
            # If any error occurs, clear and display a message
            ax.clear()
            ax.text(0.5, 0.5, f"Error creating properties table: {str(e)}",
                  ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    def visualize_torch_model(self, architecture, model_builder=None):
        """
        Visualize a neural network architecture using PyTorch and torchviz.
        This provides a more detailed internal structure view than the simplified diagrams.

        Args:
            architecture: Architecture configuration
            model_builder: Optional ModelBuilder instance to create the PyTorch model

        Returns:
            fig: Matplotlib figure with the PyTorch model visualization
        """
        if not TORCH_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "PyTorch not available for model visualization",
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax.axis('off')
            return fig

        if not TORCHVIZ_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "torchviz not available. Install with: pip install torchviz",
                ha='center',
                va='center',
                fontsize=14,
                fontweight='bold')
            ax.axis('off')
            return fig

        if model_builder is None:
            # Try to import the model builder
            try:
                from ..architecture.model_builder import ModelBuilder
                model_builder = ModelBuilder(device='cpu')
            except ImportError:
                try:
                    # Alternative approach to get the model builder
                    from snas.architecture.model_builder import ModelBuilder
                    model_builder = ModelBuilder(device='cpu')
                except ImportError:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(
                        0.5,
                        0.5,
                        "ModelBuilder not found and not provided",
                        ha='center',
                        va='center',
                        fontsize=14,
                        fontweight='bold')
                    ax.axis('off')
                    return fig

        # Get input shape and create a model
        input_shape = architecture.get('input_shape', (1, 28, 28))
        num_classes = architecture.get('num_classes', 10)

        try:
            # Build PyTorch model
            model = model_builder.build_model(architecture)
            # Switch to evaluation mode to avoid issues with batch normalization
            model.eval()

            # Create dummy input with batch size of 2 to avoid batch norm errors
            if len(input_shape) == 3:  # Image input
                x = torch.randn(2, *input_shape).to('cpu')  # Use batch size of 2 instead of 1
            else:  # Flat input
                x = torch.randn(2, input_shape[0]).to('cpu')  # Use batch size of 2 instead of 1

            # Generate graph with error handling
            with torch.no_grad():  # Disable gradient calculation for visualization
                try:
                    # Get model input/output
                    y = model(x)
                    
                    # Create a more detailed dot graph showing operations, not just parameters
                    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
                    
                    # Modify dot attributes to show more details
                    for node in dot.body:
                        if 'label=' in node:
                            # Make labels more readable by expanding truncated names
                            node = node.replace('\\n', '\\l')  # Left-align multi-line labels
                            
                            # Colorize different operations
                            if 'conv' in node.lower():
                                node = node.replace('fillcolor=lightblue', 'fillcolor=lightblue:cyan')
                            elif 'relu' in node.lower() or 'activation' in node.lower():
                                node = node.replace('fillcolor=lightblue', 'fillcolor=lightyellow')
                            elif 'pool' in node.lower():
                                node = node.replace('fillcolor=lightblue', 'fillcolor=lightgreen')
                            elif 'linear' in node.lower() or 'fc' in node.lower():
                                node = node.replace('fillcolor=lightblue', 'fillcolor=lightpink')
                                
                except ValueError as ve:
                    # If we get a batch size error, try with a larger batch
                    if "expected more than 1 value" in str(ve).lower():
                        # Try with batch size 4
                        if len(input_shape) == 3:
                            x = torch.randn(4, *input_shape).to('cpu')
                        else:
                            x = torch.randn(4, input_shape[0]).to('cpu')
                        y = model(x)
                        dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
                    else:
                        raise

            # Set rendering options with improved styling
            # Top to bottom layout with more details
            dot.attr('graph', rankdir='TB', size="12,12", 
                    concentrate='true',  # Merge multiple edges
                    nodesep='0.4',       # Space between nodes
                    ranksep='0.5')       # Space between ranks
                    
            # Create better node styling
            dot.attr('node', 
                    shape='box',         # Box shape for nodes
                    style='filled,rounded',  # Rounded corners and filled
                    fontname='Arial',    # Cleaner font
                    fontsize='12',       # Larger font size
                    height='0.4',        # Uniform height
                    penwidth='1.5')      # Thicker borders
                    
            # Improve edge styling
            dot.attr('edge',
                    fontname='Arial',    # Match node font
                    fontsize='10',       # Smaller font for edges
                    penwidth='1.3',      # Slightly thicker lines
                    arrowsize='0.8')     # Smaller arrowheads

            # Save to a temporary file
            temp_path = 'temp_model_graph'
            dot.render(temp_path, format='png')

            # Read the rendered image and create a figure
            import matplotlib.image as mpimg
            img = mpimg.imread(f'{temp_path}.png')

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(img)
            ax.axis('off')

            # Clean up temporary files
            if os.path.exists(f'{temp_path}.png'):
                os.remove(f'{temp_path}.png')
            if os.path.exists(f'{temp_path}'):
                os.remove(f'{temp_path}')
            if os.path.exists(f'{temp_path}.pdf'):
                os.remove(f'{temp_path}.pdf')

            return fig

        except Exception as e:
            # Handle errors gracefully but try a simpler textual representation as fallback
            error_msg = str(e)
            
            # Check for batch normalization specific error
            if "Expected more than 1 value per channel when training" in error_msg:
                error_msg = "BatchNorm error: Try increasing the batch size or disabling batch normalization."
            
            try:
                # Create a text-based representation of the model as a fallback
                fig, ax = plt.subplots(figsize=(10, 12), dpi=100)
                ax.axis('off')
                
                # Add error info
                ax.text(0.5, 0.98, f"Error creating graph visualization: {error_msg}",
                      ha='center', va='top', fontsize=10, color='red', wrap=True,
                      transform=ax.transAxes)
                
                # Create a simplified text visualization of the model structure
                model_str = str(model)
                # Format the model string
                model_str = model_str.replace('(', '\n  (')
                model_str = model_str.replace('): (', '):\n    (')
                
                # Add a model info header
                y_pos = 0.95
                ax.text(0.5, y_pos, "MODEL ARCHITECTURE (Simplified View)",
                      ha='center', va='top', fontsize=14, fontweight='bold',
                      transform=ax.transAxes)
                
                # Add model type
                y_pos -= 0.03
                network_type = architecture.get('network_type', 'Unknown')
                ax.text(0.5, y_pos, f"Type: {network_type.upper()}", 
                      ha='center', va='top', fontsize=12, fontweight='bold',
                      transform=ax.transAxes)
                
                # Add the model structure
                y_pos -= 0.05
                ax.text(0.05, y_pos, model_str, 
                      ha='left', va='top', fontsize=10, fontfamily='monospace',
                      transform=ax.transAxes)
                
                # Add parameter count
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                ax.text(0.5, 0.02, 
                      f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}",
                      ha='center', va='bottom', fontsize=12, fontweight='bold',
                      transform=ax.transAxes)
                
                # Add a border around the text display
                rect = Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, 
                              edgecolor='gray', linestyle='--', transform=ax.transAxes)
                ax.add_patch(rect)
                
            except:
                # If even the fallback fails, show just the error
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(
                    0.5,
                    0.5,
                    f"Error creating PyTorch model visualization: {error_msg}",
                    ha='center',
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    wrap=True,
                    transform=ax.transAxes)
                ax.axis('off')
            
            return fig

    def debug_history_data(self, history):
        """
        Debug utility to inspect and validate history data.

        Args:
            history: Search history dictionary

        Returns:
            fig: Matplotlib figure with debug information
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # Check if history is a dictionary
        if not isinstance(history, dict):
            ax.text(
                0.5,
                0.5,
                f"Error: History is not a dictionary, but {type(history)}",
                ha='center',
                va='center',
                fontsize=14,
                fontweight='bold',
                color='red')
            return fig

        # Check for required keys
        essential_keys = ['generations', 'best_fitness', 'avg_fitness']
        missing_keys = [key for key in essential_keys if key not in history]

        # Start building the debug text
        debug_text = "## History Data Debug Information\n\n"

        # Add missing keys warning if any
        if missing_keys:
            debug_text += f"[WARNING] Missing keys: {', '.join(missing_keys)}\n\n"

        # Add key info for each key in history
        debug_text += "### Content Summary:\n\n"
        for key, value in history.items():
            if isinstance(value, list):
                debug_text += f"- **{key}**: List with {len(value)} items"
                if value:
                    debug_text += f" (sample: {value[0]}, type: {type(value[0]).__name__})"
                debug_text += "\n"
            else:
                debug_text += f"- **{key}**: {type(value).__name__} = {value}\n"

        # Add validation info
        debug_text += "\n### Validation:\n\n"

        # Check consistency of generation data
        lengths = {}
        for key in [
            'generations',
            'best_fitness',
            'avg_fitness',
                'population_diversity']:
            if key in history and isinstance(history[key], list):
                lengths[key] = len(history[key])

        if lengths:
            if len(set(lengths.values())) == 1:
                debug_text += "[OK] All data arrays have the same length\n"
            else:
                debug_text += "[WARNING] Data arrays have inconsistent lengths:\n"
                for key, length in lengths.items():
                    debug_text += f"  - {key}: {length} items\n"

        # Check if there's enough data to plot
        if 'generations' in history and len(history['generations']) > 1:
            debug_text += "[OK] Enough generations for plotting\n"
        else:
            debug_text += "[WARNING] Not enough generations for meaningful plotting\n"

        # Check specific metrics
        if 'best_fitness' in history and history['best_fitness']:
            min_val = min(history['best_fitness'])
            max_val = max(history['best_fitness'])
            debug_text += f"- Best fitness range: {min_val:.4f} to {max_val:.4f}\n"

        # Add raw data section for key metrics
        for key in ['generations', 'best_fitness', 'avg_fitness']:
            if key in history and history[key]:
                debug_text += f"\n### Raw {key} data:\n{history[key]}\n"

        # Display the debug text using safe coordinates
        if debug_text:
            ax.text(
                0.05,
                0.95,
                debug_text,
                fontsize=10,
                va='top',
                ha='left',
                transform=ax.transAxes,  # Use axes coordinates for safety
                bbox=dict(
                    boxstyle='round',
                    facecolor='lightyellow',
                    alpha=0.5))

        return fig

    def plot_surrogate_accuracy(self, history, fig_size=(12, 10), save_path=None):
        """
        Plot surrogate model accuracy metrics over time.

        Args:
            history: Search history from the search algorithm
            fig_size: Size of the figure
            save_path: Path to save the figure (or None to display)

        Returns:
            matplotlib figure
        """
        if 'surrogate_accuracy' not in history or not history['surrogate_accuracy']:
            print("No surrogate accuracy data available in history")
            return None
            
        # Extract accuracy metrics
        pearson_values = [metrics.get('pearson', 0) for metrics in history['surrogate_accuracy']]
        spearman_values = [metrics.get('spearman', 0) for metrics in history['surrogate_accuracy']]
        mae_values = [metrics.get('mae', 0) for metrics in history['surrogate_accuracy']]
        sample_sizes = [metrics.get('sample_size', 0) for metrics in history['surrogate_accuracy']]
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=fig_size)
        
        # Plot correlation metrics
        x = list(range(1, len(pearson_values) + 1))
        axes[0].plot(x, pearson_values, 'o-', label='Pearson Correlation', color='#2077B4')
        axes[0].plot(x, spearman_values, 's-', label='Spearman Correlation', color='#D62728')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('Evaluation Points')
        axes[0].set_ylabel('Correlation Coefficient')
        axes[0].set_title('Surrogate Model Prediction Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE and sample size
        ax2 = axes[0].twinx()
        ax2.plot(x, mae_values, 'd-', label='Mean Absolute Error', color='#FF7F0E')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend(loc='lower right')
        
        # Plot the most recent prediction comparison
        if 'surrogate_predictions_history' in history and 'actual_performances_history' in history:
            if history['surrogate_predictions_history'] and history['actual_performances_history']:
                surrogate_preds = history['surrogate_predictions_history'][-1]
                actual_perfs = history['actual_performances_history'][-1]
                
                # Ensure equal length
                min_len = min(len(surrogate_preds), len(actual_perfs))
                surrogate_preds = surrogate_preds[:min_len]
                actual_perfs = actual_perfs[:min_len]
                
                # Sort by actual performance for better visualization
                sorted_indices = sorted(range(len(actual_perfs)), key=lambda i: actual_perfs[i])
                surrogate_sorted = [surrogate_preds[i] for i in sorted_indices]
                actual_sorted = [actual_perfs[i] for i in sorted_indices]
                
                # Plot
                axes[1].plot(actual_sorted, label='Actual Performance', marker='o', linestyle='-', color='#2077B4')
                axes[1].plot(surrogate_sorted, label='Surrogate Prediction', marker='x', linestyle='--', color='#FF7F0E')
                axes[1].set_xlabel('Architecture Index (sorted by actual performance)')
                axes[1].set_ylabel('Performance Value')
                axes[1].set_title('Comparison of Actual Performance vs. Surrogate Prediction')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Add correlation annotation
                import numpy as np
                from scipy.stats import pearsonr
                corr, _ = pearsonr(np.array(surrogate_preds), np.array(actual_perfs))
                axes[1].annotate(f'Correlation: {corr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
        
    def plot_importance_weight_impact(self, history, fig_size=(14, 10), save_path=None):
        """
        Visualize the impact of different importance weight values on hybrid performance.
        
        Args:
            history: Dictionary containing search history with importance weight data
            fig_size: Size of the figure
            save_path: Path to save the figure (or None to display)
            
        Returns:
            matplotlib figure or None if data not available
        """
        if ('importance_weight_history' not in history or not history['importance_weight_history'] or
            'surrogate_accuracy' not in history or not history['surrogate_accuracy']):
            print("No importance weight impact data available in history")
            return None
            
        # Extract data
        importance_weights = history['importance_weight_history']
        accuracy_metrics = history['surrogate_accuracy']
        
        if len(importance_weights) != len(accuracy_metrics):
            print("Mismatch between importance weights and accuracy metrics")
            return None
            
        # Prepare data for plotting
        weights = []
        pearson_values = []
        spearman_values = []
        mae_values = []
        
        for w, metrics in zip(importance_weights, accuracy_metrics):
            weights.append(w)
            pearson_values.append(metrics.get('pearson', 0))
            spearman_values.append(metrics.get('spearman', 0))
            mae_values.append(metrics.get('mae', 0))
            
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=fig_size)
        
        # Sort by weight for line plots
        sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i])
        sorted_weights = [weights[i] for i in sorted_indices]
        sorted_pearson = [pearson_values[i] for i in sorted_indices]
        sorted_spearman = [spearman_values[i] for i in sorted_indices]
        sorted_mae = [mae_values[i] for i in sorted_indices]
        
        # Plot correlation vs weight
        axes[0].plot(sorted_weights, sorted_pearson, 'o-', label='Pearson Correlation', color='#2077B4')
        axes[0].plot(sorted_weights, sorted_spearman, 's-', label='Spearman Correlation', color='#D62728')
        axes[0].set_xlabel('Shared Weights Importance (α)')
        axes[0].set_ylabel('Correlation Coefficient')
        axes[0].set_title('Impact of Importance Weight on Prediction Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE vs weight
        axes[1].plot(sorted_weights, sorted_mae, 'd-', label='Mean Absolute Error', color='#FF7F0E')
        axes[1].set_xlabel('Shared Weights Importance (α)')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Impact of Importance Weight on Prediction Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add annotations for optimal values
        best_pearson_idx = sorted_pearson.index(max(sorted_pearson))
        best_spearman_idx = sorted_spearman.index(max(sorted_spearman))
        best_mae_idx = sorted_mae.index(min(sorted_mae))
        
        axes[0].annotate(f'Best Pearson: α={sorted_weights[best_pearson_idx]:.2f}',
                      xy=(sorted_weights[best_pearson_idx], sorted_pearson[best_pearson_idx]),
                      xytext=(5, 10), textcoords='offset points',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                      
        axes[0].annotate(f'Best Spearman: α={sorted_weights[best_spearman_idx]:.2f}',
                      xy=(sorted_weights[best_spearman_idx], sorted_spearman[best_spearman_idx]),
                      xytext=(5, -20), textcoords='offset points',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                      
        axes[1].annotate(f'Best MAE: α={sorted_weights[best_mae_idx]:.2f}',
                      xy=(sorted_weights[best_mae_idx], sorted_mae[best_mae_idx]),
                      xytext=(5, 10), textcoords='offset points',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
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
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        # Encode buffer as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        return img_str
