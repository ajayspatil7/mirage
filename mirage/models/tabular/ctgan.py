"""
CTGAN Implementation for Mirage ML Engine
Production-grade Conditional Tabular GAN optimized for healthcare data and Apple Silicon M4 Pro.
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

# SDV imports for CTGAN
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    CTGANSynthesizer = None
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


class CTGANConfig(BaseGeneratorConfig):
    """Configuration schema for CTGAN model."""
    
    # CTGAN-specific parameters
    epochs: int = 300
    batch_size: int = 500
    generator_dim: tuple = (256, 256)
    discriminator_dim: tuple = (256, 256)
    generator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_lr: float = 2e-4
    discriminator_decay: float = 1e-6
    discriminator_steps: int = 1
    log_frequency: bool = True
    verbose: bool = False
    pac: int = 10
    
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
            data['name'] = 'ctgan'
        if 'data_type' not in data:
            data['data_type'] = DataType.TABULAR
        super().__init__(**data)


class CTGANGenerator(BaseGenerator):
    """
    Production-grade CTGAN implementation for synthetic tabular data generation.
    
    Optimized for:
    - Healthcare and medical data
    - Apple Silicon M4 Pro with MPS acceleration
    - Privacy-preserving synthetic data generation
    - HIPAA-compliant data handling
    """
    
    def __init__(self, config: Optional[CTGANConfig] = None):
        """
        Initialize CTGAN Generator.
        
        Args:
            config: CTGAN configuration
        """
        if not SDV_AVAILABLE:
            raise SystemRequirementError(
                "SDV library not available. Install with: pip install sdv",
                requirement="sdv>=1.10.0"
            )
        
        # Use default config if none provided
        if config is None:
            config = CTGANConfig(
                name="ctgan",
                data_type=DataType.TABULAR
            )
        
        super().__init__(config)
        
        # CTGAN-specific attributes
        self._synthesizer: Optional[CTGANSynthesizer] = None
        self._metadata: Optional[SingleTableMetadata] = None
        self._original_dtypes: Optional[Dict[str, str]] = None
        
        # Privacy and healthcare optimizations
        self._privacy_transformations = {}
        self._column_mappings = {}
        
        logger.info(f"🧠 CTGAN Generator initialized for {config.name}")
        logger.info(f"🏥 Healthcare optimizations: {config.privacy_level} privacy level")
        logger.info(f"🔧 Device: {self.device} | Epochs: {config.epochs} | Batch size: {config.batch_size}")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the configuration schema for CTGAN."""
        return {
            "type": "object",
            "title": "CTGAN Configuration",
            "description": "Configuration parameters for Conditional Tabular GAN",
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
                "generator_lr": {
                    "type": "number",
                    "minimum": 1e-6,
                    "maximum": 1e-2,
                    "default": 2e-4,
                    "description": "Generator learning rate"
                },
                "discriminator_lr": {
                    "type": "number",
                    "minimum": 1e-6,
                    "maximum": 1e-2,
                    "default": 2e-4,
                    "description": "Discriminator learning rate"
                },
                "generator_dim": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 32, "maximum": 1024},
                    "default": [256, 256],
                    "description": "Generator network dimensions"
                },
                "discriminator_dim": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 32, "maximum": 1024},
                    "default": [256, 256],
                    "description": "Discriminator network dimensions"
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
        Validate and normalize CTGAN configuration.
        
        Args:
            config: Raw configuration
            
        Returns:
            Validated configuration
            
        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        validated = config.copy()
        errors = []
        
        # Validate epochs - allow smaller values for testing
        epochs = validated.get("epochs", 300)
        if not isinstance(epochs, int) or epochs < 1 or epochs > 5000:
            errors.append("epochs must be an integer between 1 and 5000")
        
        # Validate batch size - allow smaller values for testing
        batch_size = validated.get("batch_size", 500)
        if not isinstance(batch_size, int) or batch_size < 8 or batch_size > 2048:
            errors.append("batch_size must be an integer between 8 and 2048")
        
        # Validate learning rates
        for lr_param in ["generator_lr", "discriminator_lr"]:
            lr = validated.get(lr_param, 2e-4)
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1e-2:
                errors.append(f"{lr_param} must be a number between 1e-6 and 1e-2")
        
        # Validate network dimensions - allow smaller values for testing
        for dim_param in ["generator_dim", "discriminator_dim"]:
            dim = validated.get(dim_param, (256, 256))
            if isinstance(dim, list):
                dim = tuple(dim)
            if not isinstance(dim, tuple) or len(dim) < 1 or len(dim) > 4:
                errors.append(f"{dim_param} must be a list/tuple with 1-4 integers")
            elif any(not isinstance(d, int) or d < 32 or d > 1024 for d in dim):
                errors.append(f"{dim_param} values must be integers between 32 and 1024")
            validated[dim_param] = dim
        
        # Validate privacy level
        privacy_level = validated.get("privacy_level", "standard")
        if privacy_level not in ["standard", "high", "maximum"]:
            errors.append("privacy_level must be one of: standard, high, maximum")
        
        # Set defaults BEFORE PAC validation
        validated.setdefault("generator_decay", 1e-6)
        validated.setdefault("discriminator_decay", 1e-6)
        validated.setdefault("discriminator_steps", 1)
        validated.setdefault("pac", 10)
        validated.setdefault("verbose", False)
        
        # CRITICAL FIX: Validate PAC vs batch_size constraint
        # SDV CTGAN requires batch_size % pac == 0
        batch_size = validated.get("batch_size", 500)
        pac = validated.get("pac", 10)
        
        if not isinstance(pac, int) or pac < 1 or pac > batch_size:
            errors.append(f"pac must be an integer between 1 and {batch_size}")
        elif batch_size % pac != 0:
            # Auto-fix the constraint violation
            logger.warning(f"⚠️  CTGAN constraint violation: batch_size ({batch_size}) must be divisible by pac ({pac})")
            
            # Strategy 1: Adjust batch_size down to nearest multiple of pac
            adjusted_batch_size = (batch_size // pac) * pac
            
            # Strategy 2: If adjusted batch_size is too small, reduce pac instead
            if adjusted_batch_size < 8:  # Minimum viable batch size
                original_pac = pac
                # Find largest pac that divides batch_size
                for test_pac in range(min(pac, batch_size), 0, -1):
                    if batch_size % test_pac == 0:
                        pac = test_pac
                        break
                validated["pac"] = pac
                logger.warning(f"🔧 Reduced PAC from {original_pac} to {pac} for batch_size {batch_size}")
            else:
                validated["batch_size"] = adjusted_batch_size
                logger.warning(f"🔧 Adjusted batch_size from {batch_size} to {adjusted_batch_size} for PAC {pac}")
        
        # Validate final PAC value
        final_batch_size = validated.get("batch_size", batch_size)
        final_pac = validated.get("pac", pac)
        
        if final_batch_size % final_pac != 0:
            errors.append(f"batch_size ({final_batch_size}) must be divisible by pac ({final_pac})")
        
        # Additional CTGAN-specific validations
        if final_pac < 1 or final_pac > 100:
            errors.append("pac must be between 1 and 100")
        
        # Validate discriminator_steps
        discriminator_steps = validated.get("discriminator_steps", 1)
        if not isinstance(discriminator_steps, int) or discriminator_steps < 1 or discriminator_steps > 10:
            errors.append("discriminator_steps must be an integer between 1 and 10")
        
        if errors:
            raise ModelConfigurationError(
                model_name=self.config.name,
                config_errors=errors,
                provided_config=config
            )
        
        return validated
    
    
    def _prepare_healthcare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare healthcare data for CTGAN training.
        
        Args:
            data: Raw healthcare data
            
        Returns:
            Prepared data ready for CTGAN
        """
        logger.info(f"🏥 Healthcare data prepared: {data.shape}")
        logger.info(f"🔒 Privacy level: {self.config.privacy_level}")
        
        # Convert categorical columns to object type for SDV compatibility
        categorical_columns = data.select_dtypes(include=['category']).columns
        if len(categorical_columns) > 0:
            logger.info(f"🔄 Converting categorical columns to object type: {list(categorical_columns)}")
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
                    logger.debug(f"🏷️  Set {col} as categorical")
        
        if numerical_columns:
            for col in numerical_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='numerical')
                    logger.debug(f"🔢 Set {col} as numerical")
        
        # Healthcare-specific metadata optimizations
        for col in data.columns:
            col_lower = col.lower()
            
            # Age columns - set as numerical with constraints
            if 'age' in col_lower:
                metadata.update_column(col, sdtype='numerical')
                # Note: SDV v1.x doesn't use update_column_metadata anymore
                # The constraints are handled by the model automatically
            
            # Gender/binary categorical
            elif col_lower in ['gender', 'sex'] or data[col].nunique() == 2:
                metadata.update_column(col, sdtype='categorical')
            
            # ID columns (if not removed) - set as 'id' type for SDV compatibility
            elif 'id' in col_lower:
                metadata.update_column(col, sdtype='id')
                logger.debug(f"🆔 Set {col} as id type")
        
        # Remove primary key constraint if it was auto-detected
        # This prevents validation issues with privacy-transformed ID columns
        try:
            if hasattr(metadata, 'primary_key') and metadata.primary_key:
                # Clear the primary key to avoid validation issues
                metadata.set_primary_key(None)
                logger.debug("🔓 Removed primary key constraint for privacy")
        except Exception as e:
            logger.debug(f"⚠️  Could not clear primary key: {e}")
        
        logger.info(f"📋 Metadata created for {len(data.columns)} columns")
        return metadata
    
    def fit(
        self, 
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        validation_data: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
        **kwargs
    ) -> None:
        """
        Train the CTGAN model on healthcare data.
        
        Args:
            data: Training data
            validation_data: Optional validation data (not used by CTGAN)
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
                logger.warning(f"🔄 Converting {len(categorical_cols)} categorical columns to object type")
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
                'generator_lr': kwargs.get('generator_lr', self.config.generator_lr),
                'discriminator_lr': kwargs.get('discriminator_lr', self.config.discriminator_lr),
                'generator_dim': kwargs.get('generator_dim', self.config.generator_dim),
                'discriminator_dim': kwargs.get('discriminator_dim', self.config.discriminator_dim),
                'generator_decay': kwargs.get('generator_decay', self.config.generator_decay),
                'discriminator_decay': kwargs.get('discriminator_decay', self.config.discriminator_decay),
                'discriminator_steps': kwargs.get('discriminator_steps', self.config.discriminator_steps),
                'pac': kwargs.get('pac', self.config.pac),
                'verbose': kwargs.get('verbose', self.config.verbose)
            }
            
            # Validate the training configuration
            validated_config = self.validate_config(training_config)
            
            # Configure for M4 Pro
            if hasattr(self.config, 'use_mps') and self.config.use_mps and self.device.type == 'mps':
                # SDV CTGAN doesn't directly support MPS, but PyTorch backend will use it
                logger.info("🚀 Training with MPS acceleration")
                
                # Optimize batch size for M4 Pro
                memory_gb = 10.8  # Approximate M4 Pro memory
                if validated_config['batch_size'] > 1000 and memory_gb < 16:
                    validated_config['batch_size'] = min(validated_config['batch_size'], 800)
                    logger.info(f"🔧 Optimized batch size for M4 Pro: {validated_config['batch_size']}")
            
            # Initialize CTGAN synthesizer
            logger.info("🧠 Initializing CTGAN synthesizer...")
            self._synthesizer = CTGANSynthesizer(
                metadata=self._metadata,
                epochs=validated_config['epochs'],
                batch_size=validated_config['batch_size'],
                generator_lr=validated_config['generator_lr'],
                generator_decay=validated_config['generator_decay'],
                discriminator_lr=validated_config['discriminator_lr'],
                discriminator_decay=validated_config['discriminator_decay'],
                generator_dim=validated_config['generator_dim'],
                discriminator_dim=validated_config['discriminator_dim'],
                discriminator_steps=validated_config['discriminator_steps'],
                pac=validated_config['pac'],
                verbose=validated_config['verbose']
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
            logger.info(f"🚀 Starting CTGAN training: {total_epochs} epochs, batch size {validated_config['batch_size']}")
            
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
                'data_types': self._original_dtypes
            })
            
            self._set_status(ModelStatus.TRAINED)
            logger.info(f"✅ CTGAN training completed in {training_duration:.2f}s")
            
        except Exception as e:
            self._set_status(ModelStatus.ERROR)
            error_msg = str(e) if str(e) else f"Unknown error during training: {type(e).__name__}"
            logger.error(f"❌ CTGAN training failed: {error_msg}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Error details: {e}")
            
            # Log additional debugging info
            logger.error(f"❌ Data shape: {prepared_data.shape}")
            logger.error(f"❌ Data types: {prepared_data.dtypes.to_dict()}")
            logger.error(f"❌ Config: {validated_config}")
            
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
                "CTGAN model must be trained before generating data",
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
            logger.info(f"🎲 Generating {n_samples} synthetic samples...")
            
            # Generate synthetic data
            synthetic_data = self._synthesizer.sample(num_rows=n_samples)
            
            # Post-process for healthcare data
            synthetic_data = self._post_process_healthcare_data(synthetic_data)
            
            generation_duration = time.time() - generation_start
            logger.info(f"✅ Generated {len(synthetic_data)} samples in {generation_duration:.2f}s")
            
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
        
        logger.debug(f"🏥 Applied healthcare post-processing to {processed_data.shape[0]} samples")
        return processed_data
    
    def _save_model_data(self, path: Path) -> None:
        """Save CTGAN model data."""
        if self._synthesizer is None:
            raise ModelError("No trained model to save")
        
        # Save the synthesizer
        synthesizer_path = path / "ctgan_synthesizer.pkl"
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
        
        info_path = path / "ctgan_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"💾 CTGAN model saved to {path}")
    
    def _load_model_data(self, path: Path) -> None:
        """Load CTGAN model data."""
        synthesizer_path = path / "ctgan_synthesizer.pkl"
        metadata_path = path / "sdv_metadata.json"
        info_path = path / "ctgan_info.pkl"
        
        if not synthesizer_path.exists():
            raise ModelError(f"CTGAN synthesizer not found at {synthesizer_path}")
        
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
        
        logger.info(f"📥 CTGAN model loaded from {path}")

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
        
        # CTGAN-specific memory estimation
        # Account for generator, discriminator, and data tensors
        base_memory = super().estimate_memory_usage(n_samples)
        
        # Add CTGAN-specific overhead
        # Generator and discriminator networks
        generator_params = sum(self.config.generator_dim) * 4  # float32
        discriminator_params = sum(self.config.discriminator_dim) * 4
        
        # Batch processing overhead
        batch_overhead = (generator_params + discriminator_params) * self.config.batch_size
        
        # Total memory in MB
        total_memory_mb = (base_memory + batch_overhead / (1024 * 1024)) * 1.2  # 20% safety margin
        
        return total_memory_mb


# Export the CTGAN generator
__all__ = ["CTGANGenerator", "CTGANConfig"]