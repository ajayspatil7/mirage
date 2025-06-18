"""
📤 Syntheta Mirage - API Response Schemas
=========================================

Pydantic V2 schemas for all API responses.
Comprehensive response models with metadata and status tracking.

Author: Syntheta Team
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

# Status enums
class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class HealthStatus(str, Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    SHUTDOWN = "shutdown"

# Base response schemas
class BaseResponse(BaseModel):
    """Base response with common fields"""
    model_config = ConfigDict(extra='allow')
    
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

class ErrorResponse(BaseResponse):
    """Error response schema"""
    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field(..., description="Machine-readable error code")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    traceback: Optional[str] = Field(None, description="Error traceback (debug mode only)")

# Job-related responses
class JobSubmissionResponse(BaseResponse):
    """Response for job submission"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(JobStatus.PENDING, description="Initial job status")
    estimated_duration_minutes: Optional[float] = Field(None, description="Estimated completion time")
    queue_position: Optional[int] = Field(None, description="Position in job queue")
    
class JobStatusResponse(BaseResponse):
    """Response for job status queries"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress_percentage: float = Field(0.0, ge=0, le=100, description="Job completion percentage")
    
    # Timing information
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Job duration in seconds")
    
    # Progress details
    current_step: Optional[str] = Field(None, description="Current processing step")
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    step_progress: Optional[float] = Field(None, description="Progress within current step")
    
    # Results
    output_path: Optional[str] = Field(None, description="Path to generated output")
    output_size_bytes: Optional[int] = Field(None, description="Size of output file")
    samples_generated: Optional[int] = Field(None, description="Number of samples generated")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")

class JobListResponse(BaseResponse):
    """Response for listing jobs"""
    jobs: List[JobStatusResponse] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total number of jobs")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(50, description="Number of jobs per page")
    has_next: bool = Field(False, description="Whether there are more pages")

# Generation-specific responses
class GenerationMetadata(BaseModel):
    """Metadata about generated data"""
    model_config = ConfigDict(extra='allow')
    
    # Model information
    model_type: str = Field(..., description="Type of model used")
    model_version: str = Field(..., description="Version of model used")
    
    # Generation parameters
    n_samples_requested: int = Field(..., description="Number of samples requested")
    n_samples_generated: int = Field(..., description="Number of samples actually generated")
    generation_time_seconds: float = Field(..., description="Time taken to generate data")
    
    # Data characteristics
    column_count: Optional[int] = Field(None, description="Number of columns in generated data")
    data_types: Optional[Dict[str, str]] = Field(None, description="Data types of columns")
    
    # Privacy and quality metrics
    privacy_level: str = Field(..., description="Privacy protection level used")
    quality_score: Optional[float] = Field(None, description="Quality score of generated data")
    
class GenerationResponse(BaseResponse):
    """Response for successful data generation"""
    job_id: str = Field(..., description="Job identifier")
    output_path: str = Field(..., description="Path to generated data file")
    output_format: str = Field(..., description="Format of output file")
    
    metadata: GenerationMetadata = Field(..., description="Generation metadata")
    
    # File information
    file_size_bytes: int = Field(..., description="Size of generated file")
    download_url: Optional[str] = Field(None, description="URL to download the file")
    expires_at: Optional[datetime] = Field(None, description="When the file expires")

# Data validation responses
class DataQualityReport(BaseModel):
    """Data quality assessment report"""
    model_config = ConfigDict(extra='allow')
    
    total_rows: int = Field(..., description="Total number of rows")
    total_columns: int = Field(..., description="Total number of columns")
    missing_values_count: int = Field(..., description="Total missing values")
    missing_percentage: float = Field(..., description="Percentage of missing values")
    
    quality_score: float = Field(..., ge=0, le=1, description="Overall quality score")
    
    # Column-specific information
    column_stats: Dict[str, Dict[str, Any]] = Field(..., description="Statistics per column")
    data_types: Dict[str, str] = Field(..., description="Detected data types")
    
    # Issues and recommendations
    issues: List[str] = Field(..., description="Data quality issues identified")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")

class HealthcareComplianceReport(BaseModel):
    """Healthcare compliance validation report"""
    model_config = ConfigDict(extra='allow')
    
    hipaa_compliant: bool = Field(..., description="Whether data is HIPAA compliant")
    risk_level: str = Field(..., description="Privacy risk level")
    
    # PHI detection
    potential_phi_columns: List[str] = Field(..., description="Columns that may contain PHI")
    phi_risk_score: float = Field(..., ge=0, le=1, description="PHI exposure risk score")
    
    # Compliance details
    compliance_issues: List[str] = Field(..., description="Compliance issues found")
    recommendations: List[str] = Field(..., description="Compliance recommendations")
    
    # Anonymization status
    anonymization_applied: bool = Field(..., description="Whether anonymization was applied")
    anonymization_method: Optional[str] = Field(None, description="Anonymization method used")

class DataValidationResponse(BaseResponse):
    """Response for data validation"""
    data_path: str = Field(..., description="Path to validated data")
    
    quality_report: DataQualityReport = Field(..., description="Data quality assessment")
    compliance_report: Optional[HealthcareComplianceReport] = Field(None, description="Healthcare compliance report")
    
    # Validation summary
    is_valid: bool = Field(..., description="Whether data passed validation")
    validation_errors: List[str] = Field(..., description="Validation errors found")
    warnings: List[str] = Field(..., description="Validation warnings")

# Model-related responses
class ModelInfo(BaseModel):
    """Information about a registered model"""
    model_config = ConfigDict(extra='allow')
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of model")
    data_type: str = Field(..., description="Data type this model handles")
    
    description: Optional[str] = Field(None, description="Model description")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    
    # Performance metrics
    training_accuracy: Optional[float] = Field(None, description="Training accuracy")
    validation_accuracy: Optional[float] = Field(None, description="Validation accuracy")
    generation_speed: Optional[float] = Field(None, description="Samples per second")
    
    # Model status
    is_loaded: bool = Field(False, description="Whether model is currently loaded")
    last_used: Optional[datetime] = Field(None, description="Last time model was used")
    usage_count: int = Field(0, description="Number of times model has been used")
    
    # File information
    model_size_bytes: Optional[int] = Field(None, description="Size of model file")
    created_at: datetime = Field(..., description="Model creation timestamp")
    updated_at: datetime = Field(..., description="Model last update timestamp")

class ModelListResponse(BaseResponse):
    """Response for listing available models"""
    models: List[ModelInfo] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models")
    model_types: List[str] = Field(..., description="Available model types")
    data_types: List[str] = Field(..., description="Available data types")

class ModelRegistrationResponse(BaseResponse):
    """Response for model registration"""
    model_id: str = Field(..., description="Unique model identifier")
    model_info: ModelInfo = Field(..., description="Registered model information")
    registration_time: datetime = Field(default_factory=datetime.utcnow, description="Registration timestamp")

# System status responses
class SystemInfo(BaseModel):
    """System information"""
    model_config = ConfigDict(extra='allow')
    
    # Hardware information
    device_type: str = Field(..., description="Primary compute device")
    device_name: str = Field(..., description="Device name/model")
    memory_total_gb: float = Field(..., description="Total system memory in GB")
    memory_available_gb: float = Field(..., description="Available memory in GB")
    
    # Software information
    python_version: str = Field(..., description="Python version")
    pytorch_version: str = Field(..., description="PyTorch version")
    mirage_version: str = Field(..., description="Mirage version")
    
    # Performance information
    cpu_count: int = Field(..., description="Number of CPU cores")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    mps_available: bool = Field(..., description="Whether MPS is available")

class HealthCheckResponse(BaseResponse):
    """Health check response"""
    status: HealthStatus = Field(..., description="Overall health status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    system_info: SystemInfo = Field(..., description="System information")
    
    # Component status
    components: Dict[str, HealthStatus] = Field(..., description="Status of individual components")
    
    # Performance metrics
    active_jobs: int = Field(..., description="Number of active jobs")
    queued_jobs: int = Field(..., description="Number of queued jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    
    # Resource usage
    memory_usage_percentage: float = Field(..., description="Memory usage percentage")
    cpu_usage_percentage: float = Field(..., description="CPU usage percentage")

# File operation responses
class FileUploadResponse(BaseResponse):
    """Response for file upload"""
    file_id: str = Field(..., description="Unique file identifier")
    file_path: str = Field(..., description="Path where file was stored")
    original_filename: str = Field(..., description="Original filename")
    
    file_size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    
    # Processing information
    processing_required: bool = Field(..., description="Whether file needs processing")
    estimated_processing_time_minutes: Optional[float] = Field(None, description="Estimated processing time")
    
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")

class FileDownloadResponse(BaseResponse):
    """Response for file download requests"""
    download_url: str = Field(..., description="URL to download the file")
    filename: str = Field(..., description="Filename for download")
    file_size_bytes: int = Field(..., description="File size in bytes")
    expires_at: datetime = Field(..., description="When the download URL expires")
    
# Batch operation responses
class BatchJobResponse(BaseResponse):
    """Response for batch job operations"""
    batch_id: str = Field(..., description="Unique batch identifier")
    job_ids: List[str] = Field(..., description="List of individual job IDs")
    
    total_jobs: int = Field(..., description="Total number of jobs in batch")
    completed_jobs: int = Field(0, description="Number of completed jobs")
    failed_jobs: int = Field(0, description="Number of failed jobs")
    
    overall_status: JobStatus = Field(JobStatus.PENDING, description="Overall batch status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

# Configuration responses
class ConfigValidationResponse(BaseResponse):
    """Response for configuration validation"""
    is_valid: bool = Field(..., description="Whether configuration is valid")
    validated_config: Optional[Dict[str, Any]] = Field(None, description="Validated and normalized config")
    
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Configuration suggestions")

# Export commonly used schemas
__all__ = [
    'BaseResponse',
    'ErrorResponse',
    'JobSubmissionResponse',
    'JobStatusResponse', 
    'JobListResponse',
    'GenerationResponse',
    'GenerationMetadata',
    'DataValidationResponse',
    'DataQualityReport',
    'HealthcareComplianceReport',
    'ModelInfo',
    'ModelListResponse',
    'ModelRegistrationResponse',
    'HealthCheckResponse',
    'SystemInfo',
    'FileUploadResponse',
    'FileDownloadResponse',
    'BatchJobResponse',
    'ConfigValidationResponse',
    'JobStatus',
    'HealthStatus'
]