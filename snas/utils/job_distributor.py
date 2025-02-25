"""
Job Distributor for S-NAS

This module manages the distribution of evaluation jobs across multiple GPUs.
"""

import os
import time
import threading
import queue
import logging
import torch
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

class JobDistributor:
    """Distributes model evaluation jobs across available GPU resources."""
    
    def __init__(self, num_workers=None, device_ids=None):
        """
        Initialize the job distributor.
        
        Args:
            num_workers: Number of worker threads (default: number of available GPUs)
            device_ids: Specific GPU device IDs to use (default: all available)
        """
        # Determine available GPUs
        self.available_gpus = []
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.available_gpus = list(range(num_gpus))
            logger.info(f"Found {num_gpus} available GPUs")
        else:
            logger.info("No GPUs available, using CPU")
            
        # Set worker count
        if num_workers is None:
            self.num_workers = max(1, len(self.available_gpus))
        else:
            self.num_workers = num_workers
            
        # Set device IDs
        if device_ids is not None:
            # Filter to ensure only valid GPUs are used
            self.device_ids = [d for d in device_ids if d in self.available_gpus]
            if not self.device_ids and self.available_gpus:
                logger.warning("Specified device IDs not available, using all available GPUs")
                self.device_ids = self.available_gpus
        else:
            self.device_ids = self.available_gpus
            
        # Fall back to CPU if no GPUs are available
        if not self.device_ids:
            logger.info("Using CPU for all tasks")
            self.device_ids = ['cpu']
            
        logger.info(f"Job distributor initialized with {self.num_workers} workers "
                  f"and devices: {self.device_ids}")
        
        # Initialize queues and threads
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.is_running = False
    
    def start(self):
        """Start the worker threads."""
        if self.is_running:
            logger.warning("Job distributor is already running")
            return
            
        logger.info("Starting job distributor workers")
        self.is_running = True
        
        # Start worker threads
        self.workers = []
        for i in range(self.num_workers):
            # Assign a device to each worker
            device = self.device_ids[i % len(self.device_ids)]
            device_str = f"cuda:{device}" if device != 'cpu' else 'cpu'
            
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i, device_str),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {len(self.workers)} worker threads")
    
    def stop(self):
        """Stop all worker threads."""
        if not self.is_running:
            return
            
        logger.info("Stopping job distributor")
        self.is_running = False
        
        # Add termination signals to queue
        for _ in range(len(self.workers)):
            self.job_queue.put(None)
            
        # Wait for workers to terminate
        for worker in self.workers:
            worker.join(timeout=5.0)
            
        self.workers = []
        logger.info("Job distributor stopped")
    
    def submit_job(self, job_id, job_fn, *args, **kwargs):
        """
        Submit a job to be executed by a worker.
        
        Args:
            job_id: Unique identifier for the job
            job_fn: Function to execute
            *args, **kwargs: Arguments to pass to the job function
        """
        if not self.is_running:
            self.start()
            
        logger.debug(f"Submitting job {job_id}")
        self.job_queue.put((job_id, job_fn, args, kwargs))
    
    def get_results(self, timeout=None, max_results=None):
        """
        Get completed job results.
        
        Args:
            timeout: Maximum time to wait for a result (None = wait forever)
            max_results: Maximum number of results to return (None = all available)
            
        Returns:
            list: List of (job_id, result) tuples
        """
        results = []
        
        try:
            # Try to get at least one result (may block)
            if not self.result_queue.empty() or timeout is not None:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
                self.result_queue.task_done()
            
            # Get any additional available results
            while (not self.result_queue.empty() and 
                  (max_results is None or len(results) < max_results)):
                result = self.result_queue.get_nowait()
                results.append(result)
                self.result_queue.task_done()
                
        except queue.Empty:
            # Timeout occurred, return what we have
            pass
            
        return results
    
    def wait_for_all(self, timeout=None):
        """
        Wait for all submitted jobs to complete.
        
        Args:
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            list: All job results as (job_id, result) tuples
        """
        start_time = time.time()
        results = []
        
        while True:
            # Check if we've timed out
            if timeout is not None and time.time() - start_time > timeout:
                break
                
            # Check if all jobs are done (queues are empty and no jobs in progress)
            if (self.job_queue.empty() and 
                self.job_queue.unfinished_tasks == 0 and 
                self.result_queue.unfinished_tasks == 0):
                break
                
            # Get available results
            try:
                result = self.result_queue.get(timeout=0.1)
                results.append(result)
                self.result_queue.task_done()
            except queue.Empty:
                # No results yet, continue waiting
                time.sleep(0.1)
                
        # Get any remaining results
        while not self.result_queue.empty():
            result = self.result_queue.get_nowait()
            results.append(result)
            self.result_queue.task_done()
            
        return results
    
    def _worker_loop(self, worker_id, device):
        """
        Main loop for worker threads.
        
        Args:
            worker_id: Unique ID for this worker
            device: Device to use for this worker ('cuda:X' or 'cpu')
        """
        logger.info(f"Worker {worker_id} started on device {device}")
        
        while self.is_running:
            try:
                # Get a job from the queue
                job_data = self.job_queue.get(timeout=0.5)
                
                # Check for termination signal
                if job_data is None:
                    logger.debug(f"Worker {worker_id} received termination signal")
                    self.job_queue.task_done()
                    break
                    
                # Unpack job data
                job_id, job_fn, args, kwargs = job_data
                logger.debug(f"Worker {worker_id} processing job {job_id}")
                
                # Add device to kwargs
                kwargs['device'] = device
                
                # Execute the job
                try:
                    result = job_fn(*args, **kwargs)
                    self.result_queue.put((job_id, result))
                    logger.debug(f"Worker {worker_id} completed job {job_id}")
                except Exception as e:
                    logger.error(f"Error in worker {worker_id} for job {job_id}: {e}")
                    self.result_queue.put((job_id, e))
                
                # Mark job as done
                self.job_queue.task_done()
                
            except queue.Empty:
                # No jobs available, continue
                continue
                
        logger.info(f"Worker {worker_id} stopped")

class ParallelEvaluator:
    """
    Evaluates multiple architectures in parallel using JobDistributor.
    """
    
    def __init__(self, evaluator, job_distributor=None, num_workers=None, device_ids=None):
        """
        Initialize the parallel evaluator.
        
        Args:
            evaluator: Evaluator instance for evaluating architectures
            job_distributor: JobDistributor instance (created if None)
            num_workers: Number of worker threads
            device_ids: GPU device IDs to use
        """
        self.evaluator = evaluator
        
        # Create job distributor if not provided
        if job_distributor is None:
            self.job_distributor = JobDistributor(num_workers, device_ids)
        else:
            self.job_distributor = job_distributor
    
    def evaluate_architectures(self, architectures, dataset_name, fast_mode=False):
        """
        Evaluate multiple architectures in parallel.
        
        Args:
            architectures: List of architecture configurations
            dataset_name: Name of the dataset to use
            fast_mode: If True, use a reduced evaluation protocol
            
        Returns:
            list: Evaluation results for each architecture
        """
        # Start the job distributor if not already running
        if not self.job_distributor.is_running:
            self.job_distributor.start()
        
        logger.info(f"Submitting {len(architectures)} architectures for parallel evaluation")
        
        # Submit evaluation jobs
        for i, arch in enumerate(architectures):
            self.job_distributor.submit_job(
                i, self._evaluation_job, arch, dataset_name, fast_mode
            )
        
        # Wait for all jobs to complete
        results = self.job_distributor.wait_for_all()
        
        # Sort results by job_id to maintain original order
        results.sort(key=lambda x: x[0])
        
        # Extract just the results (discard job IDs)
        return [r[1] for r in results]
    
    def _evaluation_job(self, architecture, dataset_name, fast_mode, device):
        """
        Job function for evaluating a single architecture.
        
        Args:
            architecture: Architecture configuration
            dataset_name: Name of the dataset
            fast_mode: If True, use a reduced evaluation protocol
            device: Device to use for training
            
        Returns:
            dict: Evaluation results
        """
        # Override the evaluator's device with the one assigned by the job distributor
        original_device = self.evaluator.device
        self.evaluator.device = device
        
        try:
            # Evaluate the architecture
            results = self.evaluator.evaluate(architecture, dataset_name, fast_mode)
            return results
        finally:
            # Restore original device
            self.evaluator.device = original_device