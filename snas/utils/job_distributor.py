import os
import time
import threading
import queue
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager
from typing import List, Dict, Any, Callable, Optional, Union, Tuple

logger = logging.getLogger(__name__)

@contextmanager
def use_mixed_precision():
    """Context manager for mixed precision training"""
    scaler = None
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    try:
        yield scaler
    finally:
        if scaler:
            del scaler

def setup_distributed(rank, world_size, backend='nccl', init_method='tcp://127.0.0.1:23456'):
    """
    Initialize the distributed environment.
    
    Args:
        rank: The global rank of this process
        world_size: The total number of processes
        backend: The backend to use (nccl for GPU, gloo for CPU)
        init_method: The URL to use for initializing the process group
    """
    try:
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            
        # Initialize the process group
        dist.init_process_group(
            backend=backend, 
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        logger.info(f"Initialized distributed process group: rank {rank}/{world_size}")
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}")
        raise

def cleanup_distributed():
    """Cleanup the distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")

class JobDistributor:
    """Distributes model evaluation jobs across available GPU resources."""
    
    def __init__(self, num_workers=None, device_ids=None, use_ddp=False, 
                 use_mixed_precision=False, init_method='tcp://127.0.0.1:23456'):
        """
        Initialize the job distributor.
        
        Args:
            num_workers: Number of worker threads/processes (default: number of available GPUs)
            device_ids: Specific GPU device IDs to use (default: all available)
            use_ddp: Whether to use DistributedDataParallel (requires multiple GPUs)
            use_mixed_precision: Whether to use mixed precision training with autocast
            init_method: URL for distributed initialization
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
            use_ddp = False  # Disable DDP for CPU-only
            use_mixed_precision = False  # Disable mixed precision for CPU-only
        
        # Set DDP and mixed precision flags
        self.use_ddp = use_ddp and len(self.device_ids) > 0 and self.device_ids[0] != 'cpu'
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.init_method = init_method
        
        # Log configuration
        mode_str = "DDP" if self.use_ddp else "threading"
        precision_str = "mixed precision" if self.use_mixed_precision else "full precision"
        logger.info(f"Job distributor initialized with {self.num_workers} workers "
                    f"using {mode_str} and {precision_str} on devices: {self.device_ids}")
        
        # Initialize queues and threads/processes
        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.processes = []
        self.is_running = False
        
        # For DDP, prepare mp context
        if self.use_ddp:
            self.mp_context = mp.get_context('spawn')
            self.mp_result_queue = self.mp_context.Queue()
        else:
            self.mp_context = None
            self.mp_result_queue = None
    
    def start(self):
        """Start the worker threads or processes."""
        if self.is_running:
            logger.warning("Job distributor is already running")
            return
            
        logger.info(f"Starting job distributor workers using {'DDP' if self.use_ddp else 'threading'}")
        self.is_running = True
        
        if self.use_ddp:
            # DDP uses processes instead of threads
            self._start_ddp_workers()
        else:
            # Traditional threading approach
            self._start_threaded_workers()
    
    def _start_threaded_workers(self):
        """Start traditional worker threads."""
        self.workers = []
        for i in range(self.num_workers):
            # Assign a device to each worker
            device_id = self.device_ids[i % len(self.device_ids)]
            device_str = f"cuda:{device_id}" if device_id != 'cpu' else 'cpu'
            
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i, device_str),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {len(self.workers)} worker threads")
    
    def _start_ddp_workers(self):
        """Start DDP worker processes."""
        self.processes = []
        # Prepare a queue to track initialization
        init_queue = self.mp_context.Queue()
        
        for i in range(self.num_workers):
            # For DDP, rank must match GPU index
            device_id = self.device_ids[i % len(self.device_ids)]
            if device_id == 'cpu':
                logger.warning("DDP requested but running on CPU, falling back to thread")
                continue
                
            process = self.mp_context.Process(
                target=self._ddp_worker_process,
                args=(i, device_id, self.num_workers, self.mp_result_queue, self.init_method, init_queue),
                daemon=True
            )
            process.start()
            self.processes.append(process)
        
        # Wait for all processes to initialize
        for _ in range(len(self.processes)):
            init_status = init_queue.get()
            if not init_status[0]:
                logger.error(f"Process {init_status[1]} failed to initialize: {init_status[2]}")
        
        if self.processes:
            logger.info(f"Started {len(self.processes)} DDP worker processes")
        else:
            logger.warning("Failed to start any DDP workers, falling back to threading")
            self._start_threaded_workers()
    
    def stop(self):
        """Stop all worker threads or processes."""
        if not self.is_running:
            return
            
        logger.info("Stopping job distributor")
        self.is_running = False
        
        if self.processes:
            # Stop DDP processes
            for p in self.processes:
                p.terminate()
            for p in self.processes:
                p.join(timeout=5.0)
            self.processes = []
        
        if self.workers:
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
        
        if self.use_ddp:
            # In DDP mode, we need to submit jobs to all processes
            for i, p in enumerate(self.processes):
                # Check if process is alive
                if not p.is_alive():
                    logger.warning(f"Process {i} not alive, restarting...")
                    self.start()  # This will restart all processes
                    break
            
            # Use a multiprocessing queue to submit jobs to DDP workers
            for rank in range(len(self.processes)):
                # Make a copy of kwargs to avoid sharing between processes
                kwargs_copy = kwargs.copy()
                # Add DDP specific arguments
                kwargs_copy['_ddp_rank'] = rank
                kwargs_copy['_ddp_world_size'] = len(self.processes)
                kwargs_copy['_use_mixed_precision'] = self.use_mixed_precision
                kwargs_copy['_job_id'] = job_id  # Include job_id in kwargs
                
                # Each process will pick up this job
                p = self.processes[rank]
                if p.is_alive():
                    # Create a pipe to send the job to the process
                    parent_conn, child_conn = self.mp_context.Pipe()
                    parent_conn.send((job_fn, args, kwargs_copy))
        else:
            # In threading mode, we use the queue
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
        
        if self.use_ddp:
            # Get results from multiprocessing queue
            try:
                # Try to get at least one result (may block)
                if timeout is not None:
                    try:
                        result = self.mp_result_queue.get(timeout=timeout)
                        results.append(result)
                    except queue.Empty:
                        pass
                else:
                    # No timeout specified, check if queue has items
                    try:
                        result = self.mp_result_queue.get_nowait()
                        results.append(result)
                    except queue.Empty:
                        pass
                
                # Get any additional available results
                while max_results is None or len(results) < max_results:
                    try:
                        result = self.mp_result_queue.get_nowait()
                        results.append(result)
                    except queue.Empty:
                        break
            except Exception as e:
                logger.error(f"Error getting results from MP queue: {e}")
        else:
            # Get results from threading queue
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
        
        if self.use_ddp:
            # DDP uses multiprocessing.Queue which doesn't have task tracking
            # So we need to keep track of submitted and completed jobs manually
            waiting = True
            while waiting:
                # Check if we've timed out
                if timeout is not None and time.time() - start_time > timeout:
                    logger.warning("Timed out waiting for DDP results")
                    break
                
                # Check for any available results
                try:
                    result = self.mp_result_queue.get(timeout=0.1)
                    results.append(result)
                except queue.Empty:
                    # No results available, check if all processes are idle
                    # This is an approximation since we don't have a way to check
                    # if all jobs are done in DDP mode
                    time.sleep(0.1)
                    
                # If no more results after waiting, assume all done
                if timeout is None:
                    try:
                        result = self.mp_result_queue.get(timeout=1.0)
                        results.append(result)
                    except queue.Empty:
                        waiting = False
        else:
            # Threading mode uses Queue.task_done to track completion
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
                
                # Add device and mixed precision flag to kwargs
                kwargs['device'] = device
                kwargs['_use_mixed_precision'] = self.use_mixed_precision
                
                # Execute the job
                try:
                    # Use mixed precision if enabled
                    if self.use_mixed_precision and device.startswith('cuda'):
                        with torch.cuda.amp.autocast():
                            result = job_fn(*args, **kwargs)
                    else:
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
    
    def _ddp_worker_process(self, rank, device_id, world_size, result_queue, init_method, init_queue):
        """
        Main function for DDP worker processes.
        
        Args:
            rank: Process rank
            device_id: GPU device ID
            world_size: Total number of processes
            result_queue: Queue for returning results
            init_method: URL for distributed initialization
            init_queue: Queue to report initialization status
        """
        device = f"cuda:{device_id}"
        
        # Set up process environment
        try:
            # Initialize the distributed environment
            setup_distributed(rank, world_size, init_method=init_method)
            init_queue.put((True, rank, ""))
        except Exception as e:
            logger.error(f"Failed to initialize DDP process {rank}: {e}")
            init_queue.put((False, rank, str(e)))
            return
        
        logger.info(f"DDP worker {rank} started on device {device}")
        
        try:
            # Create parent connection to receive jobs
            parent_conn, child_conn = mp.Pipe()
            
            # Process jobs until shutdown
            while True:
                # Wait for a job
                if parent_conn.poll(0.5):
                    job_fn, args, kwargs = parent_conn.recv()
                    
                    # Extract job_id from kwargs
                    job_id = kwargs.pop('_job_id', None)
                    
                    # Add device to kwargs
                    kwargs['device'] = device
                    kwargs['ddp_rank'] = rank
                    kwargs['ddp_world_size'] = world_size
                    
                    # Check for mixed precision flag
                    use_mixed_precision = kwargs.pop('_use_mixed_precision', False)
                    
                    # Execute the job
                    try:
                        # Use mixed precision if enabled
                        if use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                with use_mixed_precision() as scaler:
                                    kwargs['scaler'] = scaler
                                    result = job_fn(*args, **kwargs)
                        else:
                            result = job_fn(*args, **kwargs)
                            
                        result_queue.put((job_id, result))
                        logger.debug(f"DDP worker {rank} completed job {job_id}")
                    except Exception as e:
                        logger.error(f"Error in DDP worker {rank} for job {job_id}: {e}")
                        result_queue.put((job_id, e))
                else:
                    # No job available, check if we should continue
                    continue
                    
        except Exception as e:
            logger.error(f"Error in DDP worker {rank}: {e}")
        finally:
            # Clean up the distributed environment
            cleanup_distributed()
            logger.info(f"DDP worker {rank} stopped")

class DDPModel(nn.Module):
    """Wrapper for models to support DDP and mixed precision."""
    
    def __init__(self, model, device, ddp_rank=None, ddp_world_size=None, 
                 use_sync_bn=True, gradient_accumulation_steps=1):
        """
        Initialize the wrapped model.
        
        Args:
            model: The PyTorch model to wrap
            device: The device to place the model on
            ddp_rank: The process rank (for DDP)
            ddp_world_size: The total number of processes (for DDP)
            use_sync_bn: Whether to use SyncBatchNorm with DDP
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        super().__init__()
        self.device = device
        self.is_ddp = ddp_rank is not None and dist.is_initialized()
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        
        # Move model to device
        model = model.to(device)
        
        if self.is_ddp:
            if use_sync_bn and device.startswith('cuda'):
                # Replace BatchNorm with SyncBatchNorm for consistent stats across GPUs
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                
            # Wrap model with DDP
            self.model = DDP(model, device_ids=[int(device.split(':')[1])] if device.startswith('cuda') else None)
        else:
            self.model = model
    
    def forward(self, *args, **kwargs):
        """Forward pass that handles gradient accumulation."""
        return self.model(*args, **kwargs)
    
    def get_base_model(self):
        """Get the underlying model (without DDP wrapper)."""
        if self.is_ddp:
            return self.model.module
        return self.model

    def state_dict(self):
        """Get state dict of the base model."""
        if self.is_ddp:
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict to the base model."""
        if self.is_ddp:
            return self.model.module.load_state_dict(state_dict, strict=strict)
        return self.model.load_state_dict(state_dict, strict=strict)

class ParallelEvaluator:
    """
    Evaluates multiple architectures in parallel using JobDistributor.
    """
    
    def __init__(self, evaluator, job_distributor=None, num_workers=None, device_ids=None, 
                 use_ddp=False, use_mixed_precision=False, use_sync_bn=True,
                 gradient_accumulation_steps=1, batch_size_scaler=None):
        """
        Initialize the parallel evaluator.
        
        Args:
            evaluator: Evaluator instance for evaluating architectures
            job_distributor: JobDistributor instance (created if None)
            num_workers: Number of worker threads/processes
            device_ids: GPU device IDs to use
            use_ddp: Whether to use DistributedDataParallel
            use_mixed_precision: Whether to use mixed precision training
            use_sync_bn: Whether to use SyncBatchNorm with DDP
            gradient_accumulation_steps: Number of steps to accumulate gradients
            batch_size_scaler: Function to scale batch size based on device count
        """
        self.evaluator = evaluator
        
        # Advanced training options
        self.use_ddp = use_ddp and torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.use_sync_bn = use_sync_bn
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Batch size scaling function
        if batch_size_scaler is None:
            # Default is to scale linearly with the number of GPUs
            if self.use_ddp:
                self.batch_size_scaler = lambda base_size, world_size: base_size * world_size
            else:
                self.batch_size_scaler = lambda base_size, world_size: base_size
        else:
            self.batch_size_scaler = batch_size_scaler
        
        # Create job distributor if not provided
        if job_distributor is None:
            self.job_distributor = JobDistributor(
                num_workers=num_workers, 
                device_ids=device_ids,
                use_ddp=self.use_ddp,
                use_mixed_precision=self.use_mixed_precision
            )
        else:
            self.job_distributor = job_distributor
    
    def evaluate_architectures(self, architectures, dataset_name, fast_mode=False, **kwargs):
        """
        Evaluate multiple architectures in parallel.
        
        Args:
            architectures: List of architecture configurations
            dataset_name: Name of the dataset to use
            fast_mode: If True, use a reduced evaluation protocol
            **kwargs: Additional arguments to pass to the evaluator
            
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
                i, self._evaluation_job, arch, dataset_name, fast_mode, **kwargs
            )
        
        # Wait for all jobs to complete
        results = self.job_distributor.wait_for_all()
        
        # Sort results by job_id to maintain original order
        results.sort(key=lambda x: x[0])
        
        # Extract just the results (discard job IDs)
        return [r[1] for r in results]
    
    def _evaluation_job(self, architecture, dataset_name, fast_mode, device, 
                       ddp_rank=None, ddp_world_size=None, scaler=None, 
                       _use_mixed_precision=False, **kwargs):
        """
        Job function for evaluating a single architecture.
        
        Args:
            architecture: Architecture configuration
            dataset_name: Name of the dataset
            fast_mode: If True, use a reduced evaluation protocol
            device: Device to use for training
            ddp_rank: Rank in the DDP process group
            ddp_world_size: Total processes in the DDP group
            scaler: GradScaler for mixed precision training
            _use_mixed_precision: Whether to use mixed precision
            **kwargs: Additional arguments for evaluation
            
        Returns:
            dict: Evaluation results
        """
        # Override the evaluator's device
        original_device = self.evaluator.device
        self.evaluator.device = device
        
        # Configure for DDP and mixed precision
        is_ddp = ddp_rank is not None and ddp_world_size is not None
        use_mixed_precision = _use_mixed_precision and torch.cuda.is_available()
        
        try:
            # Add DDP and mixed precision configuration to the evaluator
            if not hasattr(self.evaluator, 'is_ddp'):
                self.evaluator.is_ddp = is_ddp
            else:
                original_is_ddp = self.evaluator.is_ddp
                self.evaluator.is_ddp = is_ddp
                
            if not hasattr(self.evaluator, 'ddp_rank'):
                self.evaluator.ddp_rank = ddp_rank
            else:
                original_ddp_rank = self.evaluator.ddp_rank
                self.evaluator.ddp_rank = ddp_rank
                
            if not hasattr(self.evaluator, 'ddp_world_size'):
                self.evaluator.ddp_world_size = ddp_world_size
            else:
                original_ddp_world_size = self.evaluator.ddp_world_size
                self.evaluator.ddp_world_size = ddp_world_size
                
            if not hasattr(self.evaluator, 'use_mixed_precision'):
                self.evaluator.use_mixed_precision = use_mixed_precision
            else:
                original_use_mixed_precision = self.evaluator.use_mixed_precision
                self.evaluator.use_mixed_precision = use_mixed_precision
                
            if not hasattr(self.evaluator, 'grad_scaler'):
                self.evaluator.grad_scaler = scaler
            else:
                original_grad_scaler = self.evaluator.grad_scaler
                self.evaluator.grad_scaler = scaler
                
            if not hasattr(self.evaluator, 'use_sync_bn'):
                self.evaluator.use_sync_bn = self.use_sync_bn
            else:
                original_use_sync_bn = self.evaluator.use_sync_bn
                self.evaluator.use_sync_bn = self.use_sync_bn
                
            if not hasattr(self.evaluator, 'gradient_accumulation_steps'):
                self.evaluator.gradient_accumulation_steps = self.gradient_accumulation_steps
            else:
                original_gradient_accumulation_steps = self.evaluator.gradient_accumulation_steps
                self.evaluator.gradient_accumulation_steps = self.gradient_accumulation_steps
                
            if not hasattr(self.evaluator, 'batch_size_scaler'):
                self.evaluator.batch_size_scaler = self.batch_size_scaler
            else:
                original_batch_size_scaler = self.evaluator.batch_size_scaler
                self.evaluator.batch_size_scaler = self.batch_size_scaler
            
            # Evaluate the architecture
            results = self.evaluator.evaluate(architecture, dataset_name, fast_mode, **kwargs)
            return results
        finally:
            # Restore original settings
            self.evaluator.device = original_device
            
            if hasattr(self, 'original_is_ddp'):
                self.evaluator.is_ddp = original_is_ddp
            if hasattr(self, 'original_ddp_rank'):
                self.evaluator.ddp_rank = original_ddp_rank
            if hasattr(self, 'original_ddp_world_size'):
                self.evaluator.ddp_world_size = original_ddp_world_size
            if hasattr(self, 'original_use_mixed_precision'):
                self.evaluator.use_mixed_precision = original_use_mixed_precision
            if hasattr(self, 'original_grad_scaler'):
                self.evaluator.grad_scaler = original_grad_scaler
            if hasattr(self, 'original_use_sync_bn'):
                self.evaluator.use_sync_bn = original_use_sync_bn
            if hasattr(self, 'original_gradient_accumulation_steps'):
                self.evaluator.gradient_accumulation_steps = original_gradient_accumulation_steps
            if hasattr(self, 'original_batch_size_scaler'):
                self.evaluator.batch_size_scaler = original_batch_size_scaler