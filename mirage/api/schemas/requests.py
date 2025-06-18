"""
🔄 Syntheta Mirage - API Request Schemas
=======================================

Pydantic V2 schemas for all API request validation.
Optimized for healthcare data generation workflows.

Author: Syntheta Team
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import PositiveInt, confloat, constr

# Data type enums
class DataType(str, Enum):
    """Supported data types for generation"""
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIMESERIES = "timeseries"

class ModelType(str, Enum):
    """Available model types"""
    CTGAN = "ctgan"
    TVAE = "tvae"
    COPULA_GAN = "copulagan"
    STABLE_DIFFUSION = "stable_diffusion"
    CUSTOM = "custom"

class PrivacyLevel(str, Enum):
    """Privacy protection levels"""
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

class OutputFormat(str, Enum):
    """Supported output formats"""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    DICOM = "dicom"
    PNG = "png"
    JPEG = "jpeg"

# Base configuration schemas
class BaseGeneratorConfig(BaseModel):
    """Base configuration for all generators"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    name: Optional[str] = Field(None, description="Custom name for this generation job")
    description: Optional[str] = Field(None, description="Description of the generation job")
    privacy_level: PrivacyLevel = Field(PrivacyLevel.STANDARD, description="Privacy protection level")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class CTGANConfig(BaseGeneratorConfig):
    """Configuration for CTGAN model"""
    epochs: int = Field(300, ge=10, le=5000, description="Number of training epochs")
    batch_size: int = Field(500, ge=50, le=2000, description="Training batch size")
    embedding_dim: int = Field(128, ge=64, le=512, description="Embedding dimension")
    generator_dim: List[int] = Field([256, 256], description="Generator layer dimensions")
    discriminator_dim: List[int] = Field([256, 256], description="Discriminator layer dimensions")
    learning_rate: float = Field(2e-4, gt=0, le=0.1, description="Learning rate")
    beta1: float = Field(0.5, ge=0, le=1, description="Adam optimizer beta1")
    beta2: float = Field(0.999, ge=0, le=1, description="Adam optimizer beta2")
    pac: int = Field(10, ge=1, le=50, description="Packing factor")
    
    @validator('generator_dim', 'discriminator_dim')
    def validate_dimensions(cls, v):
        if not v or any(dim < 64 for dim in v):
            raise ValueError("All dimensions must be >= 64")
        return v

class TVAEConfig(BaseGeneratorConfig):
    """Configuration for TVAE model"""
    epochs: int = Field(300, ge=10, le=5000, description="Number of training epochs")
    batch_size: int = Field(500, ge=50, le=2000, description="Training batch size")
    embedding_dim: int = Field(128, ge=64, le=512, description="Embedding dimension")
    compress_dims: List[int] = Field([128, 128], description="Encoder layer dimensions")
    decompress_dims: List[int] = Field([128, 128], description="Decoder layer dimensions")
    learning_rate: float = Field(1e-3, gt=0, le=0.1, description="Learning rate")
    l2scale: float = Field(1e-5, ge=0, le=1e-2, description="L2 regularization scale")

# Request schemas
class GenerateRequest(BaseModel):
    """Main request schema for synthetic data generation"""
    model_config = ConfigDict(extra='forbid')
    
    # Model specification
    model_type: ModelType = Field(..., description="Type of model to use")
    model_version: Optional[str] = Field("latest", description="Model version to use")
    
    # Data specification
    data_type: DataType = Field(DataType.TABULAR, description="Type of data to generate")
    n_samples: PositiveInt = Field(..., description="Number of samples to generate")
    
    # Configuration
    config: Optional[Dict[str, Any]] = Field(None, description="Model-specific configuration")
    
    # Data source (for training new models)
    training_data_path: Optional[str] = Field(None, description="Path to training data file")
    use_pretrained: bool = Field(True, description="Use pretrained model if available")
    
    # Output specification
    output_format: OutputFormat = Field(OutputFormat.CSV, description="Output file format")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    
    # Privacy and compliance
    privacy_level: PrivacyLevel = Field(PrivacyLevel.STANDARD, description="Privacy protection level")
    healthcare_mode: bool = Field(True, description="Enable healthcare-specific optimizations")
    
    # Job options
    priority: int = Field(5, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    timeout_minutes: int = Field(60, ge=5, le=480, description="Job timeout in minutes")
    notify_on_completion: bool = Field(False, description="Send notification when complete")
    
    @validator('config')
    def validate_config_with_model_type(cls, v, values):
        """Validate config matches model type"""
        if v is None:
            return v
            
        model_type = values.get('model_type')
        if not model_type:
            return v
            
        # Basic validation - specific validation happens in the generator
        if model_type == ModelType.CTGAN:
            # Ensure required CTGAN fields exist
            required_fields = ['epochs', 'batch_size']
            for field in required_fields:
                if field in v and not isinstance(v[field], int):
                    raise ValueError(f"CTGAN config field '{field}' must be an integer")
        
        return v
    
    @validator('n_samples')
    def validate_sample_count(cls, v):
        """Validate reasonable sample count"""
        if v > 1000000:  # 1M samples max
            raise ValueError("Cannot generate more than 1,000,000 samples in a single job")
        return v

class FileUploadRequest(BaseModel):
    """Request schema for file upload"""
    model_config = ConfigDict(extra='forbid')
    
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=1, description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    description: Optional[str] = Field(None, description="File description")
    
    # Healthcare specific
    data_type: DataType = Field(DataType.TABULAR, description="Type of data in file")
    contains_phi: bool = Field(False, description="File contains PHI/sensitive data")
    anonymization_required: bool = Field(True, description="Require data anonymization")
    
    @validator('file_size')
    def validate_file_size(cls, v):
        """Validate file size limits"""
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError(f"File size cannot exceed {max_size / 1024 / 1024:.0f}MB")
        return v
    
    @validator('content_type')
    def validate_content_type(cls, v):
        """Validate allowed content types"""
        allowed_types = [
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/json',
            'image/png',
            'image/jpeg',
            'application/dicom'
        ]
        if v not in allowed_types:
            raise ValueError(f"Content type '{v}' not supported")
        return v

class JobUpdateRequest(BaseModel):
    """Request schema for updating job properties"""
    model_config = ConfigDict(extra='forbid')
    
    priority: Optional[int] = Field(None, ge=1, le=10, description="Update job priority")
    timeout_minutes: Optional[int] = Field(None, ge=5, le=480, description="Update timeout")
    notify_on_completion: Optional[bool] = Field(None, description="Update notification setting")

class ModelRegistrationRequest(BaseModel):
    """Request schema for registering new models"""
    model_config = ConfigDict(extra='forbid')
    
    name: constr(min_length=1, max_length=100) = Field(..., description="Model name")
    version: constr(min_length=1, max_length=20) = Field(..., description="Model version")
    model_type: ModelType = Field(..., description="Type of model")
    data_type: DataType = Field(..., description="Data type this model handles")
    
    description: Optional[str] = Field(None, description="Model description")
    tags: Optional[List[str]] = Field(None, description="Model tags for categorization")
    
    # Model files
    model_file_path: str = Field(..., description="Path to model file")
    config_file_path: Optional[str] = Field(None, description="Path to config file")
    metadata_file_path: Optional[str] = Field(None, description="Path to metadata file")
    
    # Performance metrics
    training_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Training accuracy")
    validation_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Validation accuracy")
    generation_speed: Optional[float] = Field(None, gt=0, description="Samples per second")
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tag format"""
        if v:
            for tag in v:
                if not isinstance(tag, str) or len(tag) == 0:
                    raise ValueError("All tags must be non-empty strings")
        return v

class HealthcareValidationRequest(BaseModel):
    """Request schema for healthcare data validation"""
    model_config = ConfigDict(extra='forbid')
    
    data_path: str = Field(..., description="Path to data file for validation")
    validation_type: str = Field("full", description="Type of validation to perform")
    
    # Compliance checks
    check_phi: bool = Field(True, description="Check for PHI/sensitive data")
    check_quality: bool = Field(True, description="Check data quality")
    check_completeness: bool = Field(True, description="Check data completeness")
    
    # Privacy requirements
    privacy_level: PrivacyLevel = Field(PrivacyLevel.HIGH, description="Required privacy level")
    anonymization_threshold: float = Field(0.05, ge=0, le=1, description="Anonymization threshold")

# Utility schemas
class ConfigValidationRequest(BaseModel):
    """Request schema for validating model configurations"""
    model_config = ConfigDict(extra='forbid')
    
    model_type: ModelType = Field(..., description="Model type to validate config for")
    config: Dict[str, Any] = Field(..., description="Configuration to validate")

class DataPreviewRequest(BaseModel):
    """Request schema for data preview"""
    model_config = ConfigDict(extra='forbid')
    
    data_path: str = Field(..., description="Path to data file")
    sample_size: int = Field(100, ge=10, le=1000, description="Number of rows to preview")
    include_stats: bool = Field(True, description="Include basic statistics")
    include_quality_report: bool = Field(True, description="Include data quality analysis")

# Batch operation schemas
class BatchGenerateRequest(BaseModel):
    """Request schema for batch generation jobs"""
    model_config = ConfigDict(extra='forbid')
    
    jobs: List[GenerateRequest] = Field(..., min_items=1, max_items=10, description="List of generation jobs")
    batch_name: Optional[str] = Field(None, description="Name for this batch")
    sequential: bool = Field(False, description="Run jobs sequentially vs parallel")
    
    @validator('jobs')
    def validate_batch_jobs(cls, v):
        """Validate batch job constraints"""
        total_samples = sum(job.n_samples for job in v)
        if total_samples > 100000:  # 100K samples max per batch
            raise ValueError("Batch cannot generate more than 100,000 total samples")
        return v

# Export commonly used schemas
__all__ = [
    'GenerateRequest',
    'FileUploadRequest', 
    'JobUpdateRequest',
    'ModelRegistrationRequest',
    'HealthcareValidationRequest',
    'ConfigValidationRequest',
    'DataPreviewRequest',
    'BatchGenerateRequest',
    'CTGANConfig',
    'TVAEConfig',
    'DataType',
    'ModelType', 
    'PrivacyLevel',
    'OutputFormat'
]