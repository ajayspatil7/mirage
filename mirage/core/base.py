"""
Base classes and interfaces for Mirage ML generators.
Defines the standard contract all synthetic data generators must implement.
"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import warnings

import torch
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

from .config import config, device, settings
from .exceptions import (
    ModelError,
    ModelConfigurationError,
    ModelTrainingError,
    ModelGenerationError,
    DataError,
    DataValidationError,
    DeviceError
)


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    TRAINED = "trained"
    GENERATING = "generating"
    ERROR = "error"


class DataType(str, Enum):
    """Supported data types for generators."""
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIME_SERIES = "time_series"


class BaseGeneratorConfig(BaseModel):
    """Base configuration schema for all generators."""
    
    # Model identification
    name: str = Field(..., description="Model name")
    version: str = Field(default="1.0.0", description="Model version")
    description: Optional[str] = Field(default=None, description="Model description")
    
    # Data configuration
    data_type: DataType = Field(..., description="Type of data this model generates")
    
    # Training configuration
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    device: Optional[str] = Field(default=None, description="Device to use (auto-detected if None)")
    
    # Performance configuration
    batch_size: int = Field(default=32, description="Training/generation batch size", ge=1, le=1024)
    max_memory_gb: Optional[float] = Field(default=None, description="Maximum memory usage in GB")
    
    # Validation
    validate_output: bool = Field(default=True, description="Validate generated data")
    
    class Config:
        use_enum_values = True
        extra = "allow"  # Allow model-specific parameters
    
    @validator('device')
    def validate_device(cls, v):
        """Validate and normalize device specification."""
        if v is None:
            return str(device)
        
        # Normalize device strings
        v = str(v).lower()
        if v in ['auto', 'default']:
            return str(device)
        elif v in ['cpu']:
            return 'cpu'
        elif v in ['mps', 'metal']:
            if config.system.mps_available:
                return 'mps'
            else:
                warnings.warn("MPS not available, falling back to CPU")
                return 'cpu'
        elif v.startswith('cuda'):
            warnings.warn("CUDA not available on Apple Silicon, falling back to MPS/CPU")
            return str(device)
        
        return v


class ModelMetadata(BaseModel):
    """Metadata about a trained model."""
    
    # Basic info
    name: str
    version: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    # Training info
    training_duration_seconds: Optional[float] = None
    training_samples: Optional[int] = None
    training_epochs: Optional[int] = None
    
    # Model info
    model_size_mb: Optional[float] = None
    parameter_count: Optional[int] = None
    data_type: DataType
    
    # Performance metrics
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    # System info
    device_used: str
    pytorch_version: str = Field(default=torch.__version__)
    python_version: str
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingProgress(BaseModel):
    """Training progress information."""
    
    epoch: int
    total_epochs: int
    loss: Optional[float] = None
    elapsed_time: float
    estimated_remaining: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        return (self.epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0.0


class BaseGenerator(ABC):
    """
    Abstract base class for all synthetic data generators.
    
    This class defines the standard interface that all generators must implement,
    ensuring consistency across different model types (CTGAN, TVAE, Diffusion, etc.).
    """
    
    def __init__(self, config: BaseGeneratorConfig):
        """
        Initialize the generator with configuration.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        self.metadata = ModelMetadata(
            name=config.name,
            version=config.version,
            data_type=config.data_type,
            device_used=config.device or str(device),
            python_version=f"{config.python_version if hasattr(config, 'python_version') else 'unknown'}"
        )
        
        self._status = ModelStatus.UNINITIALIZED
        self._device = torch.device(config.device or device)
        self._model = None
        self._training_data_shape: Optional[Tuple[int, ...]] = None
        self._column_names: Optional[List[str]] = None
        self._training_start_time: Optional[float] = None
        
        # Setup device and optimizations
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Setup and validate the device for this generator."""
        try:
            # Test device is available
            test_tensor = torch.tensor([1.0], device=self._device)
            _ = test_tensor.sum()  # Simple operation to test device
            
            # Set thread count for optimal performance
            if self._device.type == 'cpu':
                torch.set_num_threads(settings.torch_threads)
                
        except Exception as e:
            raise DeviceError(
                message=f"Failed to setup device {self._device}",
                device=str(self._device),
                fallback_available=True,
                original_exception=e
            )
    
    @property
    def status(self) -> ModelStatus:
        """Get current model status."""
        return self._status
    
    @property
    def device(self) -> torch.device:
        """Get the device this generator is using."""
        return self._device
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._status == ModelStatus.TRAINED
    
    @property
    def training_data_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the shape of the training data."""
        return self._training_data_shape
    
    @property
    def column_names(self) -> Optional[List[str]]:
        """Get column names for tabular data."""
        return self._column_names
    
    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema for this generator.
        
        Returns:
            JSONSchema dictionary defining valid configuration parameters
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize configuration parameters.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def fit(
        self, 
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        validation_data: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> None:
        """
        Train the generator on the provided data.
        
        Args:
            data: Training data
            validation_data: Optional validation data
            **kwargs: Additional training parameters
            
        Raises:
            ModelTrainingError: If training fails
            DataValidationError: If data is invalid
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        n_samples: int,
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray, torch.Tensor]:
        """
        Generate synthetic data samples.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated synthetic data
            
        Raises:
            ModelGenerationError: If generation fails
            ModelError: If model is not trained
        """
        pass
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> None:
        """
        Validate input data format and content.
        
        Args:
            data: Data to validate
            
        Raises:
            DataValidationError: If data is invalid
        """
        if data is None:
            raise DataValidationError("Data cannot be None")
        
        # Check for empty data
        if hasattr(data, '__len__') and len(data) == 0:
            raise DataValidationError("Data cannot be empty")
        
        # Type-specific validation
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise DataValidationError("DataFrame cannot be empty")
            if data.isnull().all().any():
                raise DataValidationError("DataFrame contains columns with all null values")
                
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                raise DataValidationError("NumPy array cannot be empty")
            if np.isnan(data).all():
                raise DataValidationError("NumPy array contains only NaN values")
                
        elif isinstance(data, torch.Tensor):
            if data.numel() == 0:
                raise DataValidationError("Tensor cannot be empty")
            if torch.isnan(data).all():
                raise DataValidationError("Tensor contains only NaN values")
        
        else:
            raise DataValidationError(f"Unsupported data type: {type(data)}")
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            
        Raises:
            ModelError: If model is not trained or save fails
        """
        if not self.is_trained:
            raise ModelError(
                "Cannot save untrained model",
                model_name=self.config.name
            )
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model-specific data (implemented by subclasses)
            self._save_model_data(path)
            
            # Save metadata
            metadata_path = path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata.dict(), f, indent=2, default=str)
            
            # Save configuration
            config_path = path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.dict(), f, indent=2, default=str)
                
        except Exception as e:
            raise ModelError(
                f"Failed to save model: {e}",
                model_name=self.config.name,
                original_exception=e
            )
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
            
        Raises:
            ModelError: If load fails
        """
        path = Path(path)
        
        if not path.exists():
            raise ModelError(
                f"Model path does not exist: {path}",
                model_name=self.config.name
            )
        
        try:
            # Load metadata
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                self.metadata = ModelMetadata(**metadata_dict)
            
            # Load configuration
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                # Update current config with saved config
                for key, value in config_dict.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            # Load model-specific data (implemented by subclasses)
            self._load_model_data(path)
            
            self._status = ModelStatus.TRAINED
            
        except Exception as e:
            self._status = ModelStatus.ERROR
            raise ModelError(
                f"Failed to load model: {e}",
                model_name=self.config.name,
                original_exception=e
            )
    
    @abstractmethod
    def _save_model_data(self, path: Path) -> None:
        """Save model-specific data. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_model_data(self, path: Path) -> None:
        """Load model-specific data. Implemented by subclasses."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model info, metadata, and status
        """
        return {
            "name": self.config.name,
            "version": self.config.version,
            "status": self._status.value if hasattr(self._status, 'value') else str(self._status),
            "data_type": self.config.data_type.value if hasattr(self.config.data_type, 'value') else str(self.config.data_type),
            "device": str(self._device),
            "is_trained": self.is_trained,
            "training_data_shape": self._training_data_shape,
            "column_names": self._column_names,
            "metadata": self.metadata.dict(),
            "config": self.config.dict()
        }
    
    def estimate_memory_usage(self, n_samples: int) -> float:
        """
        Estimate memory usage for generating n_samples.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Estimated memory usage in MB
        """
        if not self._training_data_shape:
            return 0.0
        
        # Basic estimation (can be overridden by subclasses)
        bytes_per_sample = 1
        for dim in self._training_data_shape[1:]:  # Skip batch dimension
            bytes_per_sample *= dim
        
        # Assume float32 (4 bytes) + overhead
        total_bytes = n_samples * bytes_per_sample * 4 * 1.5  # 50% overhead
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _update_training_progress(self, progress: TrainingProgress) -> None:
        """Update training progress. Can be overridden for custom progress handling."""
        # Default implementation: just print progress
        if progress.epoch % 10 == 0 or progress.epoch == progress.total_epochs:
            loss_str = f"{progress.loss:.6f}" if progress.loss is not None else "N/A"
            print(f"Epoch {progress.epoch}/{progress.total_epochs} "
                  f"({progress.progress_percent:.1f}%) - "
                  f"Loss: {loss_str}")
    
    def _set_status(self, status: ModelStatus) -> None:
        """Update model status."""
        self._status = status
        self.metadata.updated_at = time.time()
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        return (f"{self.__class__.__name__}("
                f"name='{self.config.name}', "
                f"version='{self.config.version}', "
                f"status='{self._status.value}', "
                f"device='{self._device}')")


# Utility functions for generator implementations

def convert_to_tensor(
    data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert various data formats to PyTorch tensor.
    
    Args:
        data: Input data
        device: Target device
        dtype: Target data type
        
    Returns:
        PyTorch tensor on specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, pd.DataFrame):
        return torch.tensor(data.values, device=device, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=device, dtype=dtype)
    else:
        raise DataError(f"Cannot convert {type(data)} to tensor")


def convert_from_tensor(
    tensor: torch.Tensor,
    target_format: str = "numpy",
    column_names: Optional[List[str]] = None
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Convert PyTorch tensor to other formats.
    
    Args:
        tensor: Input tensor
        target_format: Target format ("numpy", "pandas")
        column_names: Column names for pandas DataFrame
        
    Returns:
        Converted data
    """
    # Move to CPU and convert to numpy
    numpy_data = tensor.detach().cpu().numpy()
    
    if target_format == "numpy":
        return numpy_data
    elif target_format == "pandas":
        if column_names and len(column_names) == numpy_data.shape[1]:
            return pd.DataFrame(numpy_data, columns=column_names)
        else:
            return pd.DataFrame(numpy_data)
    else:
        raise DataError(f"Unsupported target format: {target_format}")


# Export all public classes and functions
__all__ = [
    "BaseGenerator",
    "BaseGeneratorConfig", 
    "ModelMetadata",
    "TrainingProgress",
    "ModelStatus",
    "DataType",
    "convert_to_tensor",
    "convert_from_tensor"
]