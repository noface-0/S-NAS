"""
Dataset Registry for S-NAS

This module manages standardized datasets with consistent preprocessing for neural architecture search.
It provides a unified interface for accessing common benchmark datasets and custom datasets.
"""

import os
import torch
import torchvision
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class CSVDataset(Dataset):
    """Dataset for loading data from a CSV file."""
    
    def __init__(self, csv_file, root_dir=None, image_col='image', label_col='label', transform=None):
        """
        Args:
            csv_file: Path to the CSV file with annotations
            root_dir: Root directory with images (if None, paths in CSV are absolute)
            image_col: Column name for image paths
            label_col: Column name for labels
            transform: Optional transform to be applied on images
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        
        # Extract unique classes and create a mapping
        self.classes = sorted(self.data_frame[label_col].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path
        img_path = self.data_frame.iloc[idx][self.image_col]
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label_str = self.data_frame.iloc[idx][self.label_col]
        label = self.class_to_idx[label_str]
        
        return image, label

class FolderDataset(Dataset):
    """Dataset for loading images from a folder structure where each subfolder is a class."""
    
    def __init__(self, root_dir, transform=None, extensions=('.jpg', '.jpeg', '.png')):
        """
        Args:
            root_dir: Root directory with class subfolders
            transform: Optional transform to be applied on images
            extensions: Tuple of valid file extensions
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions
        
        # Get class folders
        self.classes = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root_dir, target_class)
            class_idx = self.class_to_idx[target_class]
            
            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if self._is_valid_file(fname):
                        path = os.path.join(root, fname)
                        self.samples.append((path, class_idx))
    
    def _is_valid_file(self, filename):
        return filename.lower().endswith(self.extensions)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path, label = self.samples[idx]
        
        # Load image
        image = Image.open(path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DatasetRegistry:
    """Registry for managing standardized datasets with consistent preprocessing."""
    
    def __init__(self, data_dir='./data', batch_size=128, num_workers=None, 
                 pin_memory=True, dataloader_kwargs=None):
        """
        Initialize the dataset registry.
        
        Args:
            data_dir: Directory to store datasets
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading (defaults to number of CPU cores)
            pin_memory: Whether to pin memory for faster GPU transfer
            dataloader_kwargs: Additional keyword arguments for DataLoader
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Set num_workers to number of CPU cores if not specified
        if num_workers is None:
            import multiprocessing
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers
            
        self.pin_memory = pin_memory
        self.dataloader_kwargs = dataloader_kwargs or {}
        
        # Custom dataset configurations
        self.custom_configs = {}
        
        # Default image transforms for different sizes
        self.transforms = {
            '32x32': {
                'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            },
            '64x64': {
                'train': transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            },
            '224x224': {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ]),
                'test': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            }
        }
        
        self.dataset_configs = {
            'cifar10': {
                'input_shape': (3, 32, 32),
                'num_classes': 10,
                'metric': 'accuracy'
            },
            'cifar100': {
                'input_shape': (3, 32, 32),
                'num_classes': 100,
                'metric': 'accuracy'
            },
            'svhn': {
                'input_shape': (3, 32, 32),
                'num_classes': 10,
                'metric': 'accuracy'
            },
            'mnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'metric': 'accuracy'
            },
            'kmnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'metric': 'accuracy'
            },
            'qmnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'metric': 'accuracy'
            },
            'emnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 47,  # Using the 'balanced' split by default
                'metric': 'accuracy'
            },
            'fashion_mnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'metric': 'accuracy'
            }
        }
    
    def get_dataset_config(self, dataset_name):
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            dict: Dataset configuration including input shape and number of classes
        """
        if dataset_name in self.dataset_configs:
            return self.dataset_configs[dataset_name]
        elif dataset_name in self.custom_configs:
            # Return a simplified config with just the required fields
            config = self.custom_configs[dataset_name]
            return {
                'input_shape': config['input_shape'],
                'num_classes': config['num_classes'],
                'metric': config['metric']
            }
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Supported datasets: {list(self.dataset_configs.keys()) + list(self.custom_configs.keys())}")
    
    def register_csv_dataset(self, name, csv_file, root_dir=None, image_col='image', 
                         label_col='label', image_size='224x224', input_shape=None, 
                         mean=None, std=None):
        """
        Register a custom dataset from a CSV file.
        
        Args:
            name: Name for the dataset
            csv_file: Path to the CSV file
            root_dir: Root directory for images (if None, paths in CSV are absolute)
            image_col: Column name for image paths
            label_col: Column name for labels
            image_size: Size of images ('32x32', '64x64', or '224x224')
            input_shape: Specify a custom input shape (optional)
            mean: Custom normalization mean (optional)
            std: Custom normalization std (optional)
            
        Returns:
            dict: Dataset configuration
        """
        # Load CSV to get class count
        df = pd.read_csv(csv_file)
        num_classes = len(df[label_col].unique())
        
        # Create custom transforms if mean and std are provided
        if mean is not None and std is not None:
            # Ensure mean and std are tuples of length 3
            if not isinstance(mean, (list, tuple)) or len(mean) != 3:
                raise ValueError("mean must be a tuple or list of length 3")
            if not isinstance(std, (list, tuple)) or len(std) != 3:
                raise ValueError("std must be a tuple or list of length 3")
                
            custom_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(int(image_size.split('x')[0])),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
                'test': transforms.Compose([
                    transforms.Resize(int(int(image_size.split('x')[0]) * 1.1)),
                    transforms.CenterCrop(int(image_size.split('x')[0])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            }
        else:
            # Use default transforms for the given image size
            if image_size not in self.transforms:
                raise ValueError(f"Unsupported image size: {image_size}. "
                               f"Supported sizes: {list(self.transforms.keys())}")
            custom_transforms = self.transforms[image_size]
        
        # Determine input shape
        if input_shape is None:
            # Default shape is (channels, height, width)
            width, height = map(int, image_size.split('x'))
            input_shape = (3, height, width)  # Assuming 3 channels (RGB)
        
        # Create dataset configuration
        config = {
            'csv_file': csv_file,
            'root_dir': root_dir,
            'image_col': image_col,
            'label_col': label_col,
            'transforms': custom_transforms,
            'input_shape': input_shape,
            'num_classes': num_classes,
            'metric': 'accuracy'
        }
        
        # Store configuration
        self.custom_configs[name] = config
        
        # Also add to dataset_configs
        self.dataset_configs[name] = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'metric': 'accuracy'
        }
        
        logger.info(f"Registered CSV dataset '{name}' with {num_classes} classes")
        return config
    
    def register_folder_dataset(self, name, root_dir, image_size='224x224', 
                              input_shape=None, mean=None, std=None,
                              extensions=('.jpg', '.jpeg', '.png')):
        """
        Register a custom dataset from a folder structure.
        
        Args:
            name: Name for the dataset
            root_dir: Root directory with class subfolders
            image_size: Size of images ('32x32', '64x64', or '224x224')
            input_shape: Specify a custom input shape (optional)
            mean: Custom normalization mean (optional)
            std: Custom normalization std (optional)
            extensions: Tuple of valid file extensions
            
        Returns:
            dict: Dataset configuration
        """
        # Get class count from folder structure
        classes = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
        num_classes = len(classes)
        
        if num_classes == 0:
            raise ValueError(f"No class folders found in {root_dir}")
        
        # Create custom transforms if mean and std are provided
        if mean is not None and std is not None:
            # Ensure mean and std are tuples of length 3
            if not isinstance(mean, (list, tuple)) or len(mean) != 3:
                raise ValueError("mean must be a tuple or list of length 3")
            if not isinstance(std, (list, tuple)) or len(std) != 3:
                raise ValueError("std must be a tuple or list of length 3")
                
            custom_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(int(image_size.split('x')[0])),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
                'test': transforms.Compose([
                    transforms.Resize(int(int(image_size.split('x')[0]) * 1.1)),
                    transforms.CenterCrop(int(image_size.split('x')[0])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            }
        else:
            # Use default transforms for the given image size
            if image_size not in self.transforms:
                raise ValueError(f"Unsupported image size: {image_size}. "
                               f"Supported sizes: {list(self.transforms.keys())}")
            custom_transforms = self.transforms[image_size]
        
        # Determine input shape
        if input_shape is None:
            # Default shape is (channels, height, width)
            width, height = map(int, image_size.split('x'))
            input_shape = (3, height, width)  # Assuming 3 channels (RGB)
        
        # Create dataset configuration
        config = {
            'root_dir': root_dir,
            'transforms': custom_transforms,
            'extensions': extensions,
            'input_shape': input_shape,
            'num_classes': num_classes,
            'metric': 'accuracy'
        }
        
        # Store configuration
        self.custom_configs[name] = config
        
        # Also add to dataset_configs
        self.dataset_configs[name] = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'metric': 'accuracy'
        }
        
        logger.info(f"Registered folder dataset '{name}' with {num_classes} classes")
        return config
    
    def get_dataset(self, dataset_name, val_split=0.1):
        """
        Get a dataset with standard preprocessing.
        
        Args:
            dataset_name: Name of the dataset
            val_split: Proportion of training data to use for validation
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Check if this is a custom dataset
        if dataset_name in self.custom_configs:
            return self._get_custom_dataset(dataset_name, val_split)
            
        # Otherwise, get a standard dataset
        if dataset_name == 'cifar10':
            return self._get_cifar10(val_split)
        elif dataset_name == 'cifar100':
            return self._get_cifar100(val_split)
        elif dataset_name == 'svhn':
            return self._get_svhn(val_split)
        elif dataset_name == 'mnist':
            return self._get_mnist(val_split)
        elif dataset_name == 'kmnist':
            return self._get_kmnist(val_split)
        elif dataset_name == 'qmnist':
            return self._get_qmnist(val_split)
        elif dataset_name == 'emnist':
            return self._get_emnist(val_split)
        elif dataset_name == 'fashion_mnist':
            return self._get_fashion_mnist(val_split)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Supported datasets: {list(self.dataset_configs.keys()) + list(self.custom_configs.keys())}")
                           
    def list_datasets(self):
        """List all available datasets."""
        standard_datasets = list(self.dataset_configs.keys())
        custom_datasets = list(self.custom_configs.keys())
        
        return {
            'standard': standard_datasets,
            'custom': custom_datasets
        }
        
    def _get_custom_dataset(self, dataset_name, val_split):
        """Load and prepare a custom dataset."""
        config = self.custom_configs[dataset_name]
        
        # Check dataset type
        if 'csv_file' in config:
            # CSV dataset
            train_set = CSVDataset(
                csv_file=config['csv_file'],
                root_dir=config['root_dir'],
                image_col=config['image_col'],
                label_col=config['label_col'],
                transform=config['transforms']['train']
            )
            
            test_set = CSVDataset(
                csv_file=config['csv_file'],
                root_dir=config['root_dir'],
                image_col=config['image_col'],
                label_col=config['label_col'],
                transform=config['transforms']['test']
            )
        else:
            # Folder dataset
            train_set = FolderDataset(
                root_dir=config['root_dir'],
                transform=config['transforms']['train'],
                extensions=config['extensions']
            )
            
            test_set = FolderDataset(
                root_dir=config['root_dir'],
                transform=config['transforms']['test'],
                extensions=config['extensions']
            )
        
        # Split dataset into train, validation, and test sets
        # For custom datasets, we use a fixed random seed for reproducibility
        dataset_size = len(train_set)
        indices = list(range(dataset_size))
        
        # Shuffle indices
        import random
        random.seed(42)
        random.shuffle(indices)
        
        # Calculate split sizes
        test_size = int(0.2 * dataset_size)  # Use 20% for test
        val_size = int(val_split * (dataset_size - test_size))  # val_split of remaining data
        train_size = dataset_size - test_size - val_size
        
        # Create subsets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_subset = Subset(train_set, train_indices)
        val_subset = Subset(test_set, val_indices)  # Use test transform for validation
        test_subset = Subset(test_set, test_indices)
        
        # Create data loaders
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_subset, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_cifar10(self, val_split):
        """Set up CIFAR-10 dataset with standard preprocessing."""
        # Define transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Load datasets
        train_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform)
        
        test_set = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_cifar100(self, val_split):
        """Set up CIFAR-100 dataset with standard preprocessing."""
        # Define transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load datasets
        train_set = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=train_transform)
        
        test_set = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=test_transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_svhn(self, val_split):
        """Set up SVHN dataset with standard preprocessing."""
        # Define transformations
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
        # Load datasets - SVHN uses 'train' and 'test' splits instead of train=True/False
        train_set = torchvision.datasets.SVHN(
            root=self.data_dir, split='train', download=True, transform=train_transform)
        
        test_set = torchvision.datasets.SVHN(
            root=self.data_dir, split='test', download=True, transform=test_transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_kmnist(self, val_split):
        """Set up KMNIST dataset with standard preprocessing."""
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1918,), (0.3483,))
        ])
        
        # Load datasets
        train_set = torchvision.datasets.KMNIST(
            root=self.data_dir, train=True, download=True, transform=transform)
        
        test_set = torchvision.datasets.KMNIST(
            root=self.data_dir, train=False, download=True, transform=transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_qmnist(self, val_split):
        """Set up QMNIST dataset with standard preprocessing."""
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Same as MNIST
        ])
        
        # Load datasets
        train_set = torchvision.datasets.QMNIST(
            root=self.data_dir, train=True, download=True, transform=transform)
        
        test_set = torchvision.datasets.QMNIST(
            root=self.data_dir, train=False, download=True, transform=transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_emnist(self, val_split):
        """Set up EMNIST dataset with standard preprocessing."""
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,))
        ])
        
        # Load datasets - Using the 'balanced' split by default (47 classes)
        train_set = torchvision.datasets.EMNIST(
            root=self.data_dir, split='balanced', train=True, download=True, transform=transform)
        
        test_set = torchvision.datasets.EMNIST(
            root=self.data_dir, split='balanced', train=False, download=True, transform=transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_mnist(self, val_split):
        """Set up MNIST dataset with standard preprocessing."""
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load datasets
        train_set = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=transform)
        
        test_set = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader
    
    def _get_fashion_mnist(self, val_split):
        """Set up Fashion-MNIST dataset with standard preprocessing."""
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        # Load datasets
        train_set = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=transform)
        
        test_set = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=transform)
        
        # Split training set into train and validation
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_subset, val_subset = random_split(
            train_set, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with configurable parameters
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        eval_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            **self.dataloader_kwargs
        }
        
        train_loader = DataLoader(train_subset, **train_loader_kwargs)
        val_loader = DataLoader(val_subset, **eval_loader_kwargs)
        test_loader = DataLoader(test_set, **eval_loader_kwargs)
        
        return train_loader, val_loader, test_loader