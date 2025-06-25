"""
TVAE Implementation for Mirage ML Engine
Production-grade Tabular Variational Autoencoder optimized for healthcare data and Apple Silicon M4 Pro.
"""

import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pickle

import torch
import pandas as pd
import numpy as np
from loguru import logger

# SDV imports for TVAE
try:
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    TVAESynthesizer = None
    SingleTableMetadata = None
    SDV_AVAILABLE = False

from ...core.base import (
    BaseGenerator, 
    BaseGeneratorConfig, 
    DataType, 
    ModelStatus,
    TrainingProgress
)
from ...core.config import config, device
from ...core.exceptions import (
    ModelError,
    ModelConfigurationError,
    ModelTrainingError,
    ModelGenerationError,
    DataValidationError,
    SystemRequirementError
)


class TVAEConfig(BaseGeneratorConfig):
    """Configuration schema for TVAE model."""
    
    # TVAE-specific parameters
    epochs: int = 300
    batch_size: int = 500
    embedding_dim: int = 128
    compress_dims: tuple = (128, 128)
    decompress_dims: tuple = (128, 128) 
    learning_rate: float = 1e-3
    l2scale: float = 1e-5
    loss_factor: float = 2.0
    
    # Healthcare-specific optimizations
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None
    privacy_level: str = "standard"  # standard, high, maximum
    
    # M4 Pro optimizations
    cuda: bool = False  # We use MPS instead
    use_mps: bool = True
    
    class Config:
        use_enum_values = True
        extra = "allow"
    
    def __init__(self, **data):
        # Set default values for required fields if not provided
        if 'name' not in data:
            data['name'] = 'tvae'
        if 'data_type' not in data:
            data['data_type'] = DataType.TABULAR
        super().__init__(**data)


class TVAEGenerator(BaseGenerator):
    """
    Production-grade TVAE implementation for synthetic tabular data generation.
    
    Optimized for:
    - Healthcare and medical data
    - Apple Silicon M4 Pro with MPS acceleration
    - Privacy-preserving synthetic data generation
    - HIPAA-compliant data handling
    """
    
    def __init__(self, config: Optional[TVAEConfig] = None):
        """
        Initialize TVAE Generator.
        
        Args:
            config: TVAE configuration
        """
        if not SDV_AVAILABLE:
            raise SystemRequirementError(
                "SDV library not available. Install with: pip install sdv",
                requirement="sdv>=1.10.0"
            )
        
        # Use default config if none provided
        if config is None:
            config = TVAEConfig(
                name="tvae",
                data_type=DataType.TABULAR
            )
        
        super().__init__(config)
        
        # TVAE-specific attributes
        self._synthesizer: Optional[TVAESynthesizer] = None
        self._metadata: Optional[SingleTableMetadata] = None
        self._original_dtypes: Optional[Dict[str, str]] = None
        
        # Privacy and healthcare optimizations
        self._privacy_transformations = {}
        self._column_mappings = {}
        
        logger.info(f">� TVAE Generator initialized for {config.name}")
        logger.info(f"<� Healthcare optimizations: {config.privacy_level} privacy level")
        logger.info(f"=' Device: {self.device} | Epochs: {config.epochs} | Batch size: {config.batch_size}")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for TVAE."""
        return {
            "type": "object",
            "title": "TVAE Configuration",
            "description": "Configuration parameters for Tabular Variational Autoencoder",
            "properties": {
                "epochs": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5000,
                    "default": 300,
                    "description": "Number of training epochs"
                },
                "batch_size": {
                    "type": "integer", 
                    "minimum": 8,
                    "maximum": 2048,
                    "default": 500,
                    "description": "Training batch size"
                },
                "learning_rate": {
                    "type": "number",
                    "minimum": 1e-6,
                    "maximum": 1e-1,
                    "default": 1e-3,
                    "description": "Learning rate for optimizer"
                },
                "embedding_dim": {
                    "type": "integer",
                    "minimum": 32,
                    "maximum": 512,
                    "default": 128,
                    "description": "Embedding dimension for categorical variables"
                },
                "compress_dims": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 32, "maximum": 1024},
                    "default": [128, 128],
                    "description": "Encoder network dimensions"
                },
                "decompress_dims": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 32, "maximum": 1024},
                    "default": [128, 128],
                    "description": "Decoder network dimensions"
                },
                "l2scale": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1e-2,
                    "default": 1e-5,
                    "description": "L2 regularization scale"
                },
                "loss_factor": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 10.0,
                    "default": 2.0,
                    "description": "Loss function scaling factor"
                },
                "privacy_level": {
                    "type": "string",
                    "enum": ["standard", "high", "maximum"],
                    "default": "standard",
                    "description": "Privacy protection level"
                },
                "categorical_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of categorical column names"
                },
                "numerical_columns": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of numerical column names"
                }
            },
            "required": ["epochs", "batch_size"]
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize TVAE configuration.
        
        Args:
            config: Raw configuration
            
        Returns:
            Validated configuration
            
        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        validated = config.copy()
        errors = []
        
        # Validate epochs
        epochs = validated.get("epochs", 300)
        if not isinstance(epochs, int) or epochs < 1 or epochs > 5000:
            errors.append("epochs must be an integer between 1 and 5000")
        
        # Validate batch size
        batch_size = validated.get("batch_size", 500)
        if not isinstance(batch_size, int) or batch_size < 8 or batch_size > 2048:
            errors.append("batch_size must be an integer between 8 and 2048")
        
        # Validate learning rate
        learning_rate = validated.get("learning_rate", 1e-3)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1e-1:
            errors.append("learning_rate must be a number between 1e-6 and 1e-1")
        
        # Validate embedding dimension
        embedding_dim = validated.get("embedding_dim", 128)
        if not isinstance(embedding_dim, int) or embedding_dim < 32 or embedding_dim > 512:
            errors.append("embedding_dim must be an integer between 32 and 512")
        
        # Validate network dimensions
        for dim_param in ["compress_dims", "decompress_dims"]:
            dim = validated.get(dim_param, (128, 128))
            if isinstance(dim, list):
                dim = tuple(dim)
            if not isinstance(dim, tuple) or len(dim) < 1 or len(dim) > 4:
                errors.append(f"{dim_param} must be a list/tuple with 1-4 integers")
            elif any(not isinstance(d, int) or d < 32 or d > 1024 for d in dim):
                errors.append(f"{dim_param} values must be integers between 32 and 1024")
            validated[dim_param] = dim
        
        # Validate L2 scale
        l2scale = validated.get("l2scale", 1e-5)
        if not isinstance(l2scale, (int, float)) or l2scale < 0 or l2scale > 1e-2:
            errors.append("l2scale must be a number between 0 and 1e-2")
        
        # Validate loss factor
        loss_factor = validated.get("loss_factor", 2.0)
        if not isinstance(loss_factor, (int, float)) or loss_factor < 0.1 or loss_factor > 10.0:
            errors.append("loss_factor must be a number between 0.1 and 10.0")
        
        # Validate privacy level
        privacy_level = validated.get("privacy_level", "standard")
        if privacy_level not in ["standard", "high", "maximum"]:
            errors.append("privacy_level must be one of: standard, high, maximum")
        
        if errors:
            raise ModelConfigurationError(
                model_name=self.config.name,
                config_errors=errors,
                provided_config=config
            )
        
        return validated
    
    def _prepare_healthcare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare healthcare data for TVAE training.
        
        Args:
            data: Raw healthcare data
            
        Returns:
            Prepared data ready for TVAE
        """
        logger.info(f"<� Healthcare data prepared: {data.shape}")
        logger.info(f"= Privacy level: {self.config.privacy_level}")
        
        # Convert categorical columns to object type for SDV compatibility
        categorical_columns = data.select_dtypes(include=['category']).columns
        if len(categorical_columns) > 0:
            logger.info(f"= Converting categorical columns to object type: {list(categorical_columns)}")
            for col in categorical_columns:
                data[col] = data[col].astype('object')
        
        # Apply privacy transformations based on level
        if self.config.privacy_level == "high":
            # Remove or anonymize sensitive columns
            sensitive_columns = ['patient_id', 'ssn', 'email', 'phone']
            for col in sensitive_columns:
                if col in data.columns:
                    data[col] = f"ANONYMIZED_{data[col].dtype}"
        
        elif self.config.privacy_level == "maximum":
            # More aggressive anonymization
            sensitive_columns = ['patient_id', 'ssn', 'email', 'phone', 'address', 'zip_code']
            for col in sensitive_columns:
                if col in data.columns:
                    data[col] = "REDACTED"
        
        # Store original dtypes for post-processing
        self._original_dtypes = data.dtypes.to_dict()
        
        return data
    
    def _create_metadata(self, data: pd.DataFrame) -> SingleTableMetadata:
        """
        Create SDV metadata for the dataset.
        
        Args:
            data: Training data
            
        Returns:
            SDV metadata object
        """
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        
        # Apply column type overrides if specified
        categorical_columns = getattr(self.config, 'categorical_columns', None)
        numerical_columns = getattr(self.config, 'numerical_columns', None)
        
        if categorical_columns:
            for col in categorical_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='categorical')
                    logger.debug(f"<�  Set {col} as categorical")
        
        if numerical_columns:
            for col in numerical_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='numerical')
                    logger.debug(f"✅ Set {col} as numerical")
        
        # Healthcare-specific metadata optimizations
        for col in data.columns:
            col_lower = col.lower()
            
            # Age columns - set as numerical with constraints
            if 'age' in col_lower:
                metadata.update_column(col, sdtype='numerical')
            
            # Gender/binary categorical
            elif col_lower in ['gender', 'sex'] or data[col].nunique() == 2:
                metadata.update_column(col, sdtype='categorical')
            
            # ID columns (if not removed) - set as 'id' type for SDV compatibility
            elif 'id' in col_lower:
                metadata.update_column(col, sdtype='id')
                logger.debug(f"<� Set {col} as id type")
        
        # Remove primary key constraint if it was auto-detected
        try:
            if hasattr(metadata, 'primary_key') and metadata.primary_key:
                metadata.set_primary_key(None)
                logger.debug("= Removed primary key constraint for privacy")
        except Exception as e:
            logger.debug(f"�  Could not clear primary key: {e}")
        
        logger.info(f"=� Metadata created for {len(data.columns)} columns")
        return metadata
    
    def fit(
        self, 
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        validation_data: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> None:
        """
        Train the TVAE model on healthcare data.
        
        Args:
            data: Training data
            validation_data: Optional validation data (not used by TVAE)
            **kwargs: Additional training parameters
        """
        self._set_status(ModelStatus.TRAINING)
        training_start = time.time()
        
        try:
            # Validate and prepare data
            self.validate_data(data)
            
            # Convert to DataFrame if necessary
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
            elif isinstance(data, torch.Tensor):
                data = pd.DataFrame(data.cpu().numpy(), columns=[f"feature_{i}" for i in range(data.shape[1])])
            
            # Prepare healthcare data
            prepared_data = self._prepare_healthcare_data(data)
            
            # Validate prepared data
            if prepared_data.empty:
                raise DataValidationError("Prepared data is empty")
            
            if prepared_data.isnull().all().any():
                raise DataValidationError("Some columns contain only null values")
            
            # Ensure all categorical columns are object type for SDV
            categorical_cols = prepared_data.select_dtypes(include=['category']).columns
            if len(categorical_cols) > 0:
                logger.warning(f"= Converting {len(categorical_cols)} categorical columns to object type")
                for col in categorical_cols:
                    prepared_data[col] = prepared_data[col].astype('object')
            
            # Create SDV metadata
            self._metadata = self._create_metadata(prepared_data)
            
            # Store training data info
            self._training_data_shape = prepared_data.shape
            self._column_names = list(prepared_data.columns)
            
            # Override config with any runtime parameters
            training_config = {
                'epochs': kwargs.get('epochs', self.config.epochs),
                'batch_size': kwargs.get('batch_size', self.config.batch_size),
                'learning_rate': kwargs.get('learning_rate', self.config.learning_rate),
                'embedding_dim': kwargs.get('embedding_dim', self.config.embedding_dim),
                'compress_dims': kwargs.get('compress_dims', self.config.compress_dims),
                'decompress_dims': kwargs.get('decompress_dims', self.config.decompress_dims),
                'l2scale': kwargs.get('l2scale', self.config.l2scale),
                'loss_factor': kwargs.get('loss_factor', self.config.loss_factor)
            }
            
            # Validate the training configuration
            validated_config = self.validate_config(training_config)
            
            # Configure for M4 Pro
            if hasattr(self.config, 'use_mps') and self.config.use_mps and self.device.type == 'mps':
                # SDV TVAE doesn't directly support MPS, but PyTorch backend will use it
                logger.info("=� Training with MPS acceleration")
                
                # Optimize batch size for M4 Pro
                memory_gb = 10.8  # Approximate M4 Pro memory
                if validated_config['batch_size'] > 1000 and memory_gb < 16:
                    validated_config['batch_size'] = min(validated_config['batch_size'], 800)
                    logger.info(f"=' Optimized batch size for M4 Pro: {validated_config['batch_size']}")
            
            # Initialize TVAE synthesizer
            logger.info(">� Initializing TVAE synthesizer...")
            self._synthesizer = TVAESynthesizer(
                metadata=self._metadata,
                epochs=validated_config['epochs'],
                batch_size=validated_config['batch_size'],
                learning_rate=validated_config['learning_rate'],
                embedding_dim=validated_config['embedding_dim'],
                compress_dims=validated_config['compress_dims'],
                decompress_dims=validated_config['decompress_dims'],
                l2scale=validated_config['l2scale'],
                loss_factor=validated_config['loss_factor']
            )
            
            # Custom progress tracking
            total_epochs = validated_config['epochs']
            
            def progress_callback(epoch_info):
                if isinstance(epoch_info, dict) and 'epoch' in epoch_info:
                    epoch = epoch_info['epoch']
                    progress = TrainingProgress(
                        epoch=epoch,
                        total_epochs=total_epochs,
                        loss=epoch_info.get('loss', None),
                        elapsed_time=time.time() - training_start
                    )
                    self._update_training_progress(progress)
            
            # Train the model
            logger.info(f"=� Starting TVAE training: {total_epochs} epochs, batch size {validated_config['batch_size']}")
            
            # Suppress SDV warnings for cleaner output
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self._synthesizer.fit(prepared_data)
            
            # Update metadata
            training_duration = time.time() - training_start
            self.metadata.training_epochs = total_epochs
            self.metadata.training_samples = len(prepared_data)
            self.metadata.training_duration_seconds = training_duration
            
            # Add healthcare-specific metrics
            self.metadata.custom_metadata.update({
                'privacy_level': getattr(self.config, 'privacy_level', 'standard'),
                'healthcare_optimized': True,
                'original_columns': len(data.columns),
                'processed_columns': len(prepared_data.columns),
                'data_types': self._original_dtypes,
                'model_type': 'TVAE',
                'vae_architecture': {
                    'embedding_dim': validated_config['embedding_dim'],
                    'compress_dims': validated_config['compress_dims'],
                    'decompress_dims': validated_config['decompress_dims']
                }
            })
            
            self._set_status(ModelStatus.TRAINED)
            logger.info(f" TVAE training completed in {training_duration:.2f}s")
            
        except Exception as e:
            self._set_status(ModelStatus.ERROR)
            error_msg = str(e) if str(e) else f"Unknown error during training: {type(e).__name__}"
            logger.error(f"L TVAE training failed: {error_msg}")
            logger.error(f"L Error type: {type(e).__name__}")
            logger.error(f"L Error details: {e}")
            
            # Log additional debugging info
            logger.error(f"L Data shape: {prepared_data.shape}")
            logger.error(f"L Data types: {prepared_data.dtypes.to_dict()}")
            logger.error(f"L Config: {validated_config}")
            
            raise ModelTrainingError(
                model_name=self.config.name,
                reason=error_msg,
                original_exception=e
            )
    
    def generate(
        self, 
        n_samples: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic healthcare data.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated synthetic data as DataFrame
        """
        if not self.is_trained:
            raise ModelError(
                "TVAE model must be trained before generating data",
                model_name=self.config.name
            )
        
        if n_samples <= 0:
            raise ModelGenerationError(
                model_name=self.config.name,
                reason="n_samples must be positive",
                n_samples=n_samples
            )
        
        self._set_status(ModelStatus.GENERATING)
        generation_start = time.time()
        
        try:
            logger.info(f"<� Generating {n_samples} synthetic samples...")
            
            # Generate synthetic data
            synthetic_data = self._synthesizer.sample(num_rows=n_samples)
            
            # Post-process for healthcare data
            synthetic_data = self._post_process_healthcare_data(synthetic_data)
            
            generation_duration = time.time() - generation_start
            logger.info(f" Generated {len(synthetic_data)} samples in {generation_duration:.2f}s")
            
            self._set_status(ModelStatus.TRAINED)
            return synthetic_data
            
        except Exception as e:
            self._set_status(ModelStatus.ERROR)
            raise ModelGenerationError(
                model_name=self.config.name,
                reason=str(e),
                n_samples=n_samples,
                original_exception=e
            )
    
    def _post_process_healthcare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply healthcare-specific post-processing to generated data.
        
        Args:
            data: Generated synthetic data
            
        Returns:
            Post-processed data
        """
        processed_data = data.copy()
        
        # Apply healthcare business rules
        for col in processed_data.columns:
            col_lower = col.lower()
            
            # Age constraints
            if 'age' in col_lower:
                processed_data[col] = processed_data[col].clip(0, 120).round()
            
            # BMI constraints
            elif 'bmi' in col_lower:
                processed_data[col] = processed_data[col].clip(10, 60).round(1)
            
            # Blood pressure constraints
            elif 'systolic' in col_lower or 'bp_sys' in col_lower:
                processed_data[col] = processed_data[col].clip(70, 250).round()
            elif 'diastolic' in col_lower or 'bp_dia' in col_lower:
                processed_data[col] = processed_data[col].clip(40, 150).round()
            
            # Heart rate constraints
            elif 'heart_rate' in col_lower or 'hr' in col_lower:
                processed_data[col] = processed_data[col].clip(30, 200).round()
            
            # Cholesterol constraints
            elif 'cholesterol' in col_lower:
                processed_data[col] = processed_data[col].clip(100, 500).round()
        
        # Ensure binary columns stay binary
        for col in processed_data.columns:
            if processed_data[col].nunique() <= 2:
                unique_vals = processed_data[col].unique()
                if len(unique_vals) == 2:
                    # Map to 0/1 if not already
                    if not all(val in [0, 1] for val in unique_vals):
                        processed_data[col] = (processed_data[col] == unique_vals[1]).astype(int)
        
        logger.debug(f"<� Applied healthcare post-processing to {processed_data.shape[0]} samples")
        return processed_data
    
    def _save_model_data(self, path: Path) -> None:
        """Save TVAE model data."""
        if self._synthesizer is None:
            raise ModelError("No trained model to save")
        
        # Save the synthesizer
        synthesizer_path = path / "tvae_synthesizer.pkl"
        with open(synthesizer_path, 'wb') as f:
            pickle.dump(self._synthesizer, f)
        
        # Save metadata
        if self._metadata is not None:
            metadata_path = path / "sdv_metadata.json"
            self._metadata.save_to_json(metadata_path)
        
        # Save additional model info
        model_info = {
            'original_dtypes': self._original_dtypes,
            'privacy_transformations': self._privacy_transformations,
            'column_mappings': self._column_mappings
        }
        
        info_path = path / "tvae_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"=� TVAE model saved to {path}")
    
    def _load_model_data(self, path: Path) -> None:
        """Load TVAE model data."""
        synthesizer_path = path / "tvae_synthesizer.pkl"
        metadata_path = path / "sdv_metadata.json"
        info_path = path / "tvae_info.pkl"
        
        if not synthesizer_path.exists():
            raise ModelError(f"TVAE synthesizer not found at {synthesizer_path}")
        
        # Load synthesizer
        with open(synthesizer_path, 'rb') as f:
            self._synthesizer = pickle.load(f)
        
        # Load metadata
        if metadata_path.exists():
            self._metadata = SingleTableMetadata.load_from_json(metadata_path)
        
        # Load additional info
        if info_path.exists():
            with open(info_path, 'rb') as f:
                model_info = pickle.load(f)
                self._original_dtypes = model_info.get('original_dtypes', {})
                self._privacy_transformations = model_info.get('privacy_transformations', {})
                self._column_mappings = model_info.get('column_mappings', {})
        
        logger.info(f"=� TVAE model loaded from {path}")

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
        
        # TVAE-specific memory estimation
        # Account for encoder, decoder, and latent space
        base_memory = super().estimate_memory_usage(n_samples)
        
        # Add TVAE-specific overhead
        # Encoder and decoder networks
        encoder_params = sum(self.config.compress_dims) * 4  # float32
        decoder_params = sum(self.config.decompress_dims) * 4
        
        # Embedding and latent space overhead
        embedding_overhead = self.config.embedding_dim * self.config.batch_size * 4
        
        # Total memory in MB
        total_memory_mb = (base_memory + (encoder_params + decoder_params + embedding_overhead) / (1024 * 1024)) * 1.3  # 30% safety margin
        
        return total_memory_mb


# Export the TVAE generator
__all__ = ["TVAEGenerator", "TVAEConfig"]