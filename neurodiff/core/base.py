"""
Base classes and interfaces for NeuroDiff components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
from torch import nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .config import DiffConfig, ModelConfig


@dataclass
class DiffResult:
    """Container for differential analysis results."""
    method: str
    layer: Optional[int] = None
    component: Optional[str] = None
    
    # Metrics
    magnitude: Optional[float] = None
    direction: Optional[torch.Tensor] = None
    
    # Detailed results
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}


class ModelWrapper(ABC):
    """Abstract base class for wrapping models for differential analysis."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.device = torch.device(config.device)
        
    @abstractmethod
    def load_model(self):
        """Load the model according to configuration."""
        pass
    
    @abstractmethod
    def get_activations(self, inputs: torch.Tensor, layer: int) -> torch.Tensor:
        """Get activations at a specific layer."""
        pass
    
    @abstractmethod
    def get_weights(self, layer: int, component: str) -> torch.Tensor:
        """Get weights for a specific component."""
        pass
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs) -> Any:
        """Forward pass through the model."""
        pass
    
    def patch_activations(self, inputs: torch.Tensor, layer: int, 
                         new_activations: torch.Tensor) -> Any:
        """Patch activations at a specific layer and continue forward pass."""
        raise NotImplementedError("Patching not implemented for this model type")


class DiffingMethod(ABC):
    """Abstract base class for different diffing methods."""
    
    def __init__(self, config: DiffConfig):
        self.config = config
        
    @abstractmethod
    def compute_diff(self, base_model: ModelWrapper, 
                    finetuned_model: ModelWrapper,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Compute the difference between models."""
        pass
    
    @abstractmethod
    def visualize(self, results: List[DiffResult]) -> Any:
        """Visualize the diffing results."""
        pass


class ActivationStore:
    """Efficient storage and retrieval of model activations."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.store = {}
        self.metadata = {}
        
    def add(self, key: str, activations: torch.Tensor, 
            metadata: Optional[Dict] = None):
        """Add activations to the store."""
        if len(self.store) >= self.max_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.store))
            del self.store[oldest_key]
            if oldest_key in self.metadata:
                del self.metadata[oldest_key]
                
        self.store[key] = activations.detach().cpu()
        if metadata:
            self.metadata[key] = metadata
            
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve activations by key."""
        return self.store.get(key)
    
    def get_batch(self, keys: List[str]) -> Optional[torch.Tensor]:
        """Retrieve a batch of activations."""
        activations = [self.get(k) for k in keys]
        if all(a is not None for a in activations):
            return torch.stack(activations)
        return None
    
    def clear(self):
        """Clear the store."""
        self.store.clear()
        self.metadata.clear()


class Hook:
    """Hook for capturing intermediate activations during forward pass."""
    
    def __init__(self, module: nn.Module, store: ActivationStore, 
                 layer_name: str):
        self.module = module
        self.store = store
        self.layer_name = layer_name
        self.handle = None
        self.enabled = True
        
    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook_fn)
        return self
        
    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()
            
    def _hook_fn(self, module, input, output):
        if self.enabled:
            # Store activations with automatic key generation
            key = f"{self.layer_name}_{id(input[0])}"
            self.store.add(key, output, {"layer": self.layer_name})
            
    def disable(self):
        self.enabled = False
        
    def enable(self):
        self.enabled = True


class MetricTracker:
    """Track and aggregate metrics during analysis."""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
        
    def update(self, name: str, value: float, step: Optional[int] = None):
        """Update a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
            self.history[name] = []
            
        self.metrics[name].append(value)
        if step is not None:
            self.history[name].append((step, value))
            
    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric."""
        if name not in self.metrics:
            return 0.0
            
        values = self.metrics[name]
        if last_n:
            values = values[-last_n:]
            
        return np.mean(values) if values else 0.0
    
    def get_all(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])
    
    def reset(self, name: Optional[str] = None):
        """Reset metrics."""
        if name:
            if name in self.metrics:
                self.metrics[name] = []
                self.history[name] = []
        else:
            self.metrics.clear()
            self.history.clear()


class CacheManager:
    """Manage caching of computed results."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, key: str, extension: str = "pt") -> Path:
        """Get path for a cache file."""
        return self.cache_dir / f"{key}.{extension}"
    
    def exists(self, key: str) -> bool:
        """Check if cache exists."""
        return any(self.cache_dir.glob(f"{key}.*"))
    
    def save(self, key: str, data: Any, format: str = "torch"):
        """Save data to cache."""
        path = self.get_cache_path(key, "pt" if format == "torch" else "pkl")
        
        if format == "torch":
            torch.save(data, path)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(data, f)
                
    def load(self, key: str, format: str = "torch") -> Any:
        """Load data from cache."""
        path = self.get_cache_path(key, "pt" if format == "torch" else "pkl")
        
        if not path.exists():
            return None
            
        if format == "torch":
            return torch.load(path, map_location="cpu")
        else:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)


class BatchProcessor:
    """Process data in batches for efficiency."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        
    def process_batches(self, data: List[Any], 
                       process_fn: callable,
                       show_progress: bool = True) -> List[Any]:
        """Process data in batches."""
        from tqdm import tqdm
        
        results = []
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(data), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches", 
                           total=num_batches)
            
        for i in iterator:
            batch = data[i:i + self.batch_size]
            batch_results = process_fn(batch)
            results.extend(batch_results)
            
        return results