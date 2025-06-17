"""
Custom exception classes for Mirage ML Engine.
Provides structured error handling for all ML pipeline components.
"""

from typing import Any, Dict, Optional, Union
import traceback


class MirageException(Exception):
    """Base exception class for all Mirage-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MIRAGE_ERROR",
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }
        
        if self.original_exception:
            result["original_error"] = {
                "type": type(self.original_exception).__name__,
                "message": str(self.original_exception)
            }
            
        return result


# =============================================================================
# Configuration and System Exceptions
# =============================================================================

class ConfigurationError(MirageException):
    """Raised when there's a configuration problem."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details=details,
            **kwargs
        )


class SystemRequirementError(MirageException):
    """Raised when system requirements are not met."""
    
    def __init__(self, message: str, requirement: Optional[str] = None, **kwargs):
        details = {"requirement": requirement} if requirement else {}
        super().__init__(
            message=message,
            error_code="SYSTEM_REQUIREMENT_ERROR",
            details=details,
            **kwargs
        )


class DeviceError(MirageException):
    """Raised when there are device/hardware related issues."""
    
    def __init__(
        self, 
        message: str, 
        device: Optional[str] = None,
        fallback_available: bool = False,
        **kwargs
    ):
        details = {
            "device": device,
            "fallback_available": fallback_available
        }
        super().__init__(
            message=message,
            error_code="DEVICE_ERROR",
            details=details,
            **kwargs
        )


# =============================================================================
# Model-Related Exceptions
# =============================================================================

class ModelError(MirageException):
    """Base class for model-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        details = {"model_name": model_name} if model_name else {}
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            details=details,
            **kwargs
        )


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    
    def __init__(
        self, 
        model_name: str, 
        version: Optional[str] = None,
        available_models: Optional[list] = None,
        **kwargs
    ):
        message = f"Model '{model_name}' not found"
        if version:
            message += f" (version: {version})"
            
        details = {
            "model_name": model_name,
            "version": version,
            "available_models": available_models or []
        }
        
        # Don't pass error_code to super() since ModelError already sets one
        super().__init__(
            message=message,
            model_name=model_name,
            **kwargs
        )
        # Override the error_code after initialization
        self.error_code = "MODEL_NOT_FOUND"
        self.details.update(details)


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    
    def __init__(
        self,
        model_name: str,
        reason: str,
        model_path: Optional[str] = None,
        **kwargs
    ):
        message = f"Failed to load model '{model_name}': {reason}"
        details = {
            "model_name": model_name,
            "reason": reason,
            "model_path": model_path
        }
        
        super().__init__(
            message=message,
            model_name=model_name,
            **kwargs
        )
        self.error_code = "MODEL_LOAD_ERROR"
        self.details.update(details)


class ModelConfigurationError(ModelError):
    """Raised when model configuration is invalid."""
    
    def __init__(
        self,
        model_name: str,
        config_errors: Union[str, list],
        provided_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if isinstance(config_errors, list):
            reason = "; ".join(config_errors)
        else:
            reason = config_errors
            
        message = f"Invalid configuration for model '{model_name}': {reason}"
        details = {
            "model_name": model_name,
            "config_errors": config_errors,
            "provided_config": provided_config
        }
        
        super().__init__(
            message=message,
            model_name=model_name,
            **kwargs
        )
        self.error_code = "MODEL_CONFIG_ERROR"
        self.details.update(details)


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    
    def __init__(
        self,
        model_name: str,
        reason: str,
        epoch: Optional[int] = None,
        **kwargs
    ):
        message = f"Training failed for model '{model_name}': {reason}"
        if epoch is not None:
            message += f" (epoch {epoch})"
            
        details = {
            "model_name": model_name,
            "reason": reason,
            "epoch": epoch
        }
        
        super().__init__(
            message=message,
            model_name=model_name,
            **kwargs
        )
        self.error_code = "MODEL_TRAINING_ERROR"
        self.details.update(details)


class ModelGenerationError(ModelError):
    """Raised when model generation fails."""
    
    def __init__(
        self,
        model_name: str,
        reason: str,
        n_samples: Optional[int] = None,
        **kwargs
    ):
        message = f"Generation failed for model '{model_name}': {reason}"
        if n_samples:
            message += f" (requested {n_samples} samples)"
            
        details = {
            "model_name": model_name,
            "reason": reason,
            "n_samples": n_samples
        }
        
        super().__init__(
            message=message,
            model_name=model_name,
            **kwargs
        )
        self.error_code = "MODEL_GENERATION_ERROR"
        self.details.update(details)


# =============================================================================
# Data-Related Exceptions
# =============================================================================

class DataError(MirageException):
    """Base class for data-related errors."""
    
    def __init__(self, message: str, data_path: Optional[str] = None, **kwargs):
        details = {"data_path": data_path} if data_path else {}
        super().__init__(
            message=message,
            error_code="DATA_ERROR",
            details=details,
            **kwargs
        )


class DataNotFoundError(DataError):
    """Raised when required data is not found."""
    
    def __init__(self, data_path: str, **kwargs):
        message = f"Data not found: {data_path}"
        super().__init__(
            message=message,
            data_path=data_path,
            **kwargs
        )
        self.error_code = "DATA_NOT_FOUND"


class DataValidationError(DataError):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[list] = None,
        data_path: Optional[str] = None,
        **kwargs
    ):
        details = {
            "validation_errors": validation_errors or [],
            "data_path": data_path
        }
        
        super().__init__(
            message=message,
            data_path=data_path,
            **kwargs
        )
        self.error_code = "DATA_VALIDATION_ERROR"
        self.details.update(details)


class DataFormatError(DataError):
    """Raised when data format is not supported or invalid."""
    
    def __init__(
        self,
        data_path: str,
        expected_format: str,
        actual_format: Optional[str] = None,
        **kwargs
    ):
        message = f"Invalid data format for {data_path}. Expected: {expected_format}"
        if actual_format:
            message += f", Got: {actual_format}"
            
        details = {
            "data_path": data_path,
            "expected_format": expected_format,
            "actual_format": actual_format
        }
        
        super().__init__(
            message=message,
            data_path=data_path,
            **kwargs
        )
        self.error_code = "DATA_FORMAT_ERROR"
        self.details.update(details)


class DataPreprocessingError(DataError):
    """Raised when data preprocessing fails."""
    
    def __init__(
        self,
        message: str,
        preprocessing_step: Optional[str] = None,
        data_path: Optional[str] = None,
        **kwargs
    ):
        details = {
            "preprocessing_step": preprocessing_step,
            "data_path": data_path
        }
        
        super().__init__(
            message=message,
            data_path=data_path,
            **kwargs
        )
        self.error_code = "DATA_PREPROCESSING_ERROR"
        self.details.update(details)


# =============================================================================
# Job and Execution Exceptions
# =============================================================================

class JobError(MirageException):
    """Base class for job execution errors."""
    
    def __init__(self, message: str, job_id: Optional[str] = None, **kwargs):
        details = {"job_id": job_id} if job_id else {}
        super().__init__(
            message=message,
            error_code="JOB_ERROR",
            details=details,
            **kwargs
        )


class JobNotFoundError(JobError):
    """Raised when a job is not found."""
    
    def __init__(self, job_id: str, **kwargs):
        message = f"Job not found: {job_id}"
        super().__init__(
            message=message,
            job_id=job_id,
            **kwargs
        )
        self.error_code = "JOB_NOT_FOUND"


class JobExecutionError(JobError):
    """Raised when job execution fails."""
    
    def __init__(
        self,
        job_id: str,
        reason: str,
        stage: Optional[str] = None,
        **kwargs
    ):
        message = f"Job execution failed: {reason}"
        if stage:
            message += f" (stage: {stage})"
            
        details = {
            "job_id": job_id,
            "reason": reason,
            "stage": stage
        }
        
        super().__init__(
            message=message,
            job_id=job_id,
            **kwargs
        )
        self.error_code = "JOB_EXECUTION_ERROR"
        self.details.update(details)


class JobTimeoutError(JobError):
    """Raised when a job times out."""
    
    def __init__(
        self,
        job_id: str,
        timeout_seconds: int,
        **kwargs
    ):
        message = f"Job {job_id} timed out after {timeout_seconds} seconds"
        details = {
            "job_id": job_id,
            "timeout_seconds": timeout_seconds
        }
        
        super().__init__(
            message=message,
            job_id=job_id,
            **kwargs
        )
        self.error_code = "JOB_TIMEOUT_ERROR"
        self.details.update(details)


class ResourceError(MirageException):
    """Raised when there are resource-related issues."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        available: Optional[float] = None,
        required: Optional[float] = None,
        **kwargs
    ):
        details = {
            "resource_type": resource_type,
            "available": available,
            "required": required
        }
        
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            details=details,
            **kwargs
        )


# =============================================================================
# API and Validation Exceptions
# =============================================================================

class APIError(MirageException):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 500, **kwargs):
        self.status_code = status_code
        super().__init__(
            message=message,
            error_code="API_ERROR",
            **kwargs
        )


class ValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        details = {"field_errors": field_errors or {}}
        super().__init__(
            message=message,
            status_code=422,
            **kwargs
        )
        self.error_code = "VALIDATION_ERROR"
        self.details.update(details)


# =============================================================================
# Utility Functions
# =============================================================================

def handle_exception(exc: Exception) -> MirageException:
    """Convert any exception to a MirageException with context."""
    if isinstance(exc, MirageException):
        return exc
    
    # Try to determine the best MirageException type
    exc_name = type(exc).__name__.lower()
    
    if "file" in exc_name or "path" in exc_name:
        return DataNotFoundError(
            data_path=str(exc),
            original_exception=exc
        )
    elif "memory" in exc_name or "cuda" in exc_name or "mps" in exc_name:
        return DeviceError(
            message=f"Device error: {exc}",
            original_exception=exc
        )
    elif "config" in exc_name or "setting" in exc_name:
        return ConfigurationError(
            message=f"Configuration error: {exc}",
            original_exception=exc
        )
    else:
        # Generic fallback
        return MirageException(
            message=f"Unexpected error: {exc}",
            error_code="UNKNOWN_ERROR",
            original_exception=exc
        )


def get_exception_context() -> Dict[str, Any]:
    """Get current exception context for debugging."""
    return {
        "traceback": traceback.format_exc(),
        "exception_type": type(Exception).__name__
    }


# =============================================================================
# Exception Registry
# =============================================================================

# Export all exception classes for easy importing
__all__ = [
    # Base
    "MirageException",
    
    # System
    "ConfigurationError",
    "SystemRequirementError", 
    "DeviceError",
    
    # Models
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelConfigurationError",
    "ModelTrainingError",
    "ModelGenerationError",
    
    # Data
    "DataError",
    "DataNotFoundError",
    "DataValidationError",
    "DataFormatError",
    "DataPreprocessingError",
    
    # Jobs
    "JobError",
    "JobNotFoundError",
    "JobExecutionError",
    "JobTimeoutError",
    "ResourceError",
    
    # API
    "APIError",
    "ValidationError",
    
    # Utilities
    "handle_exception",
    "get_exception_context"
]