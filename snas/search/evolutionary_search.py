"""
Evolutionary Search for S-NAS

This module implements an evolutionary algorithm to search for optimal neural
network architectures in the defined architecture space.
"""

import random
import time
import json
import logging
import numpy as np
from typing import List, Dict, Callable, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class EvolutionarySearch:
    """Evolutionary algorithm for neural architecture search."""
    
    def __init__(self, architecture_space, evaluator, dataset_name, 
                 population_size=20, mutation_rate=0.2, crossover_rate=0.5,
                 generations=10, elite_size=2, tournament_size=3,
                 metric='val_acc', save_history=True):
        """
        Initialize the evolutionary search.
        
        Args:
            architecture_space: Space of possible architectures
            evaluator: Component to evaluate architectures
            dataset_name: Name of the dataset to use
            population_size: Size of the population
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover vs. mutation
            generations: Number of generations to evolve
            elite_size: Number of top individuals to preserve unchanged
            tournament_size: Size of tournaments for parent selection
            metric: Metric to optimize ('val_acc', 'test_acc', etc.)
            save_history: Whether to save the full history
        """
        self.architecture_space = architecture_space
        self.evaluator = evaluator
        self.dataset_name = dataset_name
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.metric = metric
        self.save_history = save_history
        
        # Initialize empty population and history
        self.population = []
        self.fitness_scores = []
        self.best_architecture = None
        self.best_fitness = float('-inf')
        
        # For tracking progress
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_architecture': [],
            'population_diversity': [],
            'evaluation_times': []
        }
        
        # For tracking all evaluated architectures to avoid duplicates
        self.evaluated_architectures = {}
    
    def initialize_population(self):
        """Create initial random population of architectures."""
        logger.info("Initializing population with %d individuals", self.population_size)
        
        self.population = []
        while len(self.population) < self.population_size:
            # Generate a random architecture
            architecture = self.architecture_space.sample_random_architecture()
            
            # Validate the architecture
            if self.architecture_space.validate_architecture(architecture):
                # Add to population
                self.population.append(architecture)
        
        logger.info("Finished initializing population")
    
    def evaluate_population(self, fast_mode=False):
        """
        Evaluate the fitness of each architecture in the population.
        
        Args:
            fast_mode: If True, use a reduced evaluation protocol for faster iterations
            
        Returns:
            list: Fitness scores for each architecture
        """
        logger.info("Evaluating population fitness")
        
        fitness_scores = []
        start_time = time.time()
        
        for i, architecture in enumerate(self.population):
            # Check if we've evaluated this architecture before
            arch_str = json.dumps(architecture, sort_keys=True)
            if arch_str in self.evaluated_architectures:
                # Use cached result
                evaluation = self.evaluated_architectures[arch_str]
                logger.info(f"Architecture {i+1}/{len(self.population)} already evaluated, using cached result")
            else:
                # Evaluate the architecture
                logger.info(f"Evaluating architecture {i+1}/{len(self.population)}")
                try:
                    evaluation = self.evaluator.evaluate(
                        architecture, self.dataset_name, fast_mode=fast_mode
                    )
                    # Cache the result
                    self.evaluated_architectures[arch_str] = evaluation
                except Exception as e:
                    logger.error(f"Error evaluating architecture: {e}")
                    # Assign a very low fitness
                    fitness_scores.append(float('-inf'))
                    continue
            
            # Get the fitness metric (default is validation accuracy)
            if self.metric == 'val_acc':
                fitness = evaluation.get('best_val_acc', 0.0)
            elif self.metric == 'test_acc':
                fitness = evaluation.get('test_acc', 0.0)
            else:
                fitness = evaluation.get(self.metric, 0.0)
            
            fitness_scores.append(fitness)
            
            # Update best architecture if better
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_architecture = architecture.copy()
                logger.info(f"New best architecture found! Fitness: {fitness:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        
        self.fitness_scores = fitness_scores
        return fitness_scores
    
    def _select_parents(self, population, fitness_scores):
        """
        Select parents using tournament selection.
        
        Args:
            population: List of architectures
            fitness_scores: Fitness score for each architecture
            
        Returns:
            list: Selected parent architectures
        """
        selected_parents = []
        
        # Select population_size parents
        for _ in range(self.population_size):
            # Randomly select tournament_size candidates
            candidate_indices = random.sample(range(len(population)), self.tournament_size)
            candidate_fitness = [fitness_scores[i] for i in candidate_indices]
            
            # Select the best candidate from the tournament
            winner_idx = candidate_indices[candidate_fitness.index(max(candidate_fitness))]
            selected_parents.append(population[winner_idx])
            
        return selected_parents
    
    def _crossover(self, parent1, parent2):
        """
        Combine two parent architectures to create a child.
        
        Args:
            parent1: First parent architecture
            parent2: Second parent architecture
            
        Returns:
            dict: Child architecture
        """
        child = {}
        
        # First, check if network types are the same
        if parent1['network_type'] != parent2['network_type']:
            # Network types are different - use one parent as the base
            # and just apply mutation to it rather than trying to combine incompatible architectures
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
        
        # For each parameter in the architecture
        for key in parent1.keys():
            # Skip input shape and num_classes, they should stay the same
            if key in ['input_shape', 'num_classes']:
                child[key] = parent1[key]
                continue
                
            # Only attempt to crossover parameters that exist in both parents
            if key not in parent2:
                child[key] = parent1[key]
                continue
                
            if isinstance(parent1[key], list) and len(parent1[key]) > 1:
                # For list parameters (like filters per layer), perform crossover
                # at a random point in the list
                crossover_point = random.randint(1, len(parent1[key])-1)
                child[key] = parent1[key][:crossover_point] + parent2[key][crossover_point:]
            else:
                # For scalar parameters, randomly choose from either parent
                child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        
        # Make sure num_layers matches the length of layer lists
        if 'num_layers' in child:
            list_params = []
            if child['network_type'] == 'mlp':
                list_params = ['hidden_units', 'activations']
            else:  # cnn, resnet, mobilenet
                list_params = ['filters', 'kernel_sizes', 'activations', 'use_skip_connections']
                
            for param in list_params:
                if param in child:
                    child['num_layers'] = len(child[param])
                    break
                
        return child
    
    def create_next_generation(self):
        """
        Create the next generation through elitism, crossover, and mutation.
        
        Returns:
            list: New population
        """
        # Sort population by fitness
        sorted_indices = sorted(range(len(self.fitness_scores)), 
                               key=lambda i: self.fitness_scores[i], reverse=True)
        
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [self.fitness_scores[i] for i in sorted_indices]
        
        new_population = []
        
        # Elitism: keep the best individuals unchanged
        for i in range(min(self.elite_size, len(sorted_population))):
            new_population.append(sorted_population[i])
        
        # Fill the rest of the population with offspring
        parents = self._select_parents(sorted_population, sorted_fitness)
        
        while len(new_population) < self.population_size:
            # Decide whether to use crossover or just mutation
            if random.random() < self.crossover_rate:
                # Select two parents and create offspring via crossover
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
            else:
                # Just select one parent
                parent = random.choice(parents)
                child = parent.copy()
            
            # Apply random mutations
            child = self.architecture_space.mutate_architecture(child, self.mutation_rate)
            
            # Validate the child
            if self.architecture_space.validate_architecture(child):
                new_population.append(child)
        
        return new_population
    
    def calculate_diversity(self):
        """
        Calculate population diversity based on parameter distributions.
        
        Returns:
            float: Diversity score (higher is more diverse)
        """
        # Count occurrences of each parameter value
        value_counts = defaultdict(lambda: defaultdict(int))
        
        for arch in self.population:
            for param, value in arch.items():
                if isinstance(value, list):
                    # For list parameters, count each element separately
                    for i, v in enumerate(value):
                        value_counts[f"{param}_{i}"][str(v)] += 1
                else:
                    # For scalar parameters
                    value_counts[param][str(value)] += 1
        
        # Calculate entropy for each parameter
        entropies = []
        for param, counts in value_counts.items():
            total = sum(counts.values())
            entropy = 0
            for count in counts.values():
                p = count / total
                entropy -= p * (p > 0 and np.log2(p) or 0)  # Shannon entropy
            entropies.append(entropy)
        
        # Average entropy across all parameters
        if entropies:
            return sum(entropies) / len(entropies)
        return 0.0
    
    def evolve(self, fast_mode_generations=0):
        """
        Run the evolutionary search process.
        
        Args:
            fast_mode_generations: Number of initial generations to run in fast mode
                                   for quicker exploration
        
        Returns:
            dict: The best architecture found
        """
        import numpy as np
        
        logger.info("Starting evolutionary search")
        self.initialize_population()
        
        for generation in range(self.generations):
            logger.info(f"Generation {generation+1}/{self.generations}")
            
            # Use fast mode for the first few generations to explore the space quickly
            use_fast_mode = generation < fast_mode_generations
            if use_fast_mode:
                logger.info("Using fast evaluation mode")
            
            # Evaluate fitness of each architecture
            fitness_scores = self.evaluate_population(fast_mode=use_fast_mode)
            
            # Record statistics for this generation
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            max_fitness = max(fitness_scores)
            best_idx = fitness_scores.index(max_fitness)
            best_arch = self.population[best_idx].copy()
            diversity = self.calculate_diversity()
            
            logger.info(f"Generation stats: Avg fitness: {avg_fitness:.4f}, "
                       f"Best fitness: {max_fitness:.4f}, Diversity: {diversity:.4f}")
            
            # Save history
            if self.save_history:
                self.history['generations'].append(generation)
                self.history['best_fitness'].append(max_fitness)
                self.history['avg_fitness'].append(avg_fitness)
                self.history['best_architecture'].append(best_arch)
                self.history['population_diversity'].append(diversity)
            
            # Create next generation (except for the last iteration)
            if generation < self.generations - 1:
                self.population = self.create_next_generation()
        
        # Final evaluation of the best architecture (if we used fast mode)
        if fast_mode_generations > 0 and self.best_architecture:
            logger.info("Performing final evaluation of best architecture")
            
            # Evaluate with full protocol
            try:
                evaluation = self.evaluator.evaluate(
                    self.best_architecture, self.dataset_name, fast_mode=False
                )
                
                # Update best fitness with full evaluation result
                if self.metric == 'val_acc':
                    self.best_fitness = evaluation.get('best_val_acc', 0.0)
                elif self.metric == 'test_acc':
                    self.best_fitness = evaluation.get('test_acc', 0.0)
                else:
                    self.best_fitness = evaluation.get(self.metric, 0.0)
                
                logger.info(f"Final evaluation result: {self.best_fitness:.4f}")
            except Exception as e:
                logger.error(f"Error in final evaluation: {e}")
        
        logger.info("Evolutionary search completed")
        return self.best_architecture, self.best_fitness, self.history