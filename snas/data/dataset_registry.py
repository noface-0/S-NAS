import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

class DatasetRegistry:
    """Registry for managing standardized datasets with consistent preprocessing."""
    
    def __init__(self, data_dir='./data', batch_size=128, num_workers=4):
        """
        Initialize the dataset registry.
        
        Args:
            data_dir: Directory to store datasets
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_configs = {
            'cifar10': {
                'input_shape': (3, 32, 32),
                'num_classes': 10,
                'metric': 'accuracy'
            },
            'mnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 10,
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
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Supported datasets: {list(self.dataset_configs.keys())}")
        
        return self.dataset_configs[dataset_name]
    
    def get_dataset(self, dataset_name, val_split=0.1):
        """
        Get a dataset with standard preprocessing.
        
        Args:
            dataset_name: Name of the dataset
            val_split: Proportion of training data to use for validation
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if dataset_name == 'cifar10':
            return self._get_cifar10(val_split)
        elif dataset_name == 'mnist':
            return self._get_mnist(val_split)
        elif dataset_name == 'fashion_mnist':
            return self._get_fashion_mnist(val_split)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Supported datasets: {list(self.dataset_configs.keys())}")
    
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
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
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
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
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
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader