"""
🎲 Syntheta Mirage - Generation API Endpoints
=============================================

FastAPI endpoints for synthetic data generation.
Integrates SerenityEngine, CTGAN, and Data Loader systems.

Author: Syntheta Team
"""

import asyncio
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, status
from fastapi.responses import FileResponse
from loguru import logger

from ...core.config import config, settings
from ...core.exceptions import (
    MirageException,
    ModelNotFoundError,
    ModelTrainingError,
    ModelGenerationError,
    DataValidationError,
    DataNotFoundError,
    JobExecutionError,
    APIError,
    ValidationError
)
from ...engine.serenity_engine import SerenityEngine, GenerationJob, JobPriority
from ...engine.queue import JobQueue, JobStatus
from ...models.tabular.ctgan import CTGANGenerator, CTGANConfig
from ...data.loaders import CSVDataLoader, HealthcareDataLoader, PrivacyLevel, MissingValueStrategy
from ..schemas.requests import (
    GenerateRequest,
    FileUploadRequest,
    DataPreviewRequest,
    ConfigValidationRequest,
    HealthcareValidationRequest,
    BatchGenerateRequest,
    ModelType,
    DataType,
    OutputFormat
)
from ..schemas.responses import (
    JobSubmissionResponse,
    JobStatusResponse,
    JobListResponse,
    GenerationResponse,
    GenerationMetadata,
    DataValidationResponse,
    DataQualityReport,
    HealthcareComplianceReport,
    FileUploadResponse,
    FileDownloadResponse,
    ConfigValidationResponse,
    BatchJobResponse,
    BaseResponse,
    ErrorResponse
)

# Initialize router
router = APIRouter(prefix="/generate", tags=["Generation"])

# Global instances
serenity_engine = SerenityEngine()

# Note: Models are created dynamically per request, not pre-registered

# Dependencies
def get_serenity_engine() -> SerenityEngine:
    """Dependency to get SerenityEngine instance"""
    return serenity_engine

# Helper functions
async def create_generator_from_request(request: GenerateRequest) -> Any:
    """Create appropriate generator based on request"""
    if request.model_type == ModelType.CTGAN:
        # Create CTGAN configuration with required fields
        config_dict = request.config or {}
        config_dict.update({
            'name': request.model_type.value,
            'data_type': DataType.TABULAR
        })
        
        ctgan_config = CTGANConfig(**config_dict)
        generator = CTGANGenerator(ctgan_config)
        return generator
    
    elif request.model_type == ModelType.TVAE:
        # TODO: Implement TVAE generator
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="TVAE model not yet implemented"
        )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model type: {request.model_type}"
        )

async def load_and_preprocess_data(data_path: str, privacy_level: PrivacyLevel = PrivacyLevel.STANDARD) -> Any:
    """Load and preprocess data from file path"""
    try:
        # Use healthcare data loader for enhanced processing
        loader = HealthcareDataLoader(
            privacy_level=privacy_level,
            missing_strategy=MissingValueStrategy.MEDIAN_IMPUTE,
            healthcare_mode=True
        )
        
        # Load the data
        raw_data = loader.load_csv(data_path)
        
        # Preprocess the data
        processed_data = loader.preprocess_data(raw_data)
        
        # Ensure categorical columns are object dtype for SDV compatibility
        for col in processed_data.columns:
            if processed_data[col].dtype.name == 'category':
                processed_data[col] = processed_data[col].astype('object')
                logger.debug(f"🔄 Converted {col} from category to object for SDV compatibility")
        
        logger.info(f"✅ Data loaded and preprocessed: {processed_data.shape}")
        return processed_data, loader
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        raise DataValidationError(f"Failed to load data from {data_path}: {str(e)}")

async def execute_generation_job(job_id: str, request: GenerateRequest, engine: SerenityEngine) -> Dict[str, Any]:
    """Execute synthetic data generation job"""
    try:
        logger.info(f"🚀 Starting generation job {job_id}")
        
        # Get the job from SerenityEngine
        job = engine.get_job(job_id)
        if not job:
            raise JobExecutionError(job_id, f"Job {job_id} not found")
        
        # Update job status to running
        job.status = JobStatus.RUNNING
        job.progress = 10
        
        # Create generator
        generator = await create_generator_from_request(request)
        logger.info(f"✅ Generator created: {type(generator).__name__}")
        
        # Update progress
        job.progress = 20
        
        # Load and preprocess data if training is needed
        if request.training_data_path and not request.use_pretrained:
            logger.info(f"📊 Loading training data from: {request.training_data_path}")
            
            processed_data, data_loader = await load_and_preprocess_data(
                request.training_data_path,
                request.privacy_level
            )
            
            # Update progress
            job.progress = 40
            
            # Train the model
            logger.info(f"🏋️ Training model...")
            generator.fit(processed_data)
            
            # Update progress
            job.progress = 70
            
        else:
            logger.info(f"📦 Using pretrained model")
            # TODO: Load pretrained model
            
        # Generate synthetic data
        logger.info(f"🎲 Generating {request.n_samples} synthetic samples...")
        
        synthetic_data = generator.generate(request.n_samples)
        
        # Update progress
        job.progress = 90
        
        # Save output file
        output_filename = request.output_filename or f"synthetic_data_{job_id}"
        output_path = config.storage_paths["outputs"] / f"{output_filename}.{request.output_format.value}"
        
        # Save based on output format
        if request.output_format == OutputFormat.CSV:
            synthetic_data.to_csv(output_path, index=False)
        elif request.output_format == OutputFormat.PARQUET:
            synthetic_data.to_parquet(output_path, index=False)
        elif request.output_format == OutputFormat.JSON:
            synthetic_data.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {request.output_format}")
        
        logger.info(f"💾 Output saved to: {output_path}")
        
        # Create generation metadata
        metadata = GenerationMetadata(
            model_type=request.model_type.value,
            model_version=request.model_version or "latest",
            n_samples_requested=request.n_samples,
            n_samples_generated=len(synthetic_data),
            generation_time_seconds=0,  # TODO: Calculate actual time
            column_count=len(synthetic_data.columns),
            data_types={col: str(dtype) for col, dtype in synthetic_data.dtypes.items()},
            privacy_level=request.privacy_level.value,
            quality_score=0.95  # TODO: Calculate actual quality score
        )
        
        # Complete the job
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.completed_at = time.time()
        
        # Store results in job
        job.output_path = str(output_path)
        job.result = {
            "job_id": job_id,
            "output_path": str(output_path),
            "output_format": request.output_format.value,
            "metadata": metadata.model_dump(),
            "file_size_bytes": output_path.stat().st_size
        }
        
        logger.info(f"✅ Generation job {job_id} completed successfully")
        return job.result
        
    except Exception as e:
        error_msg = f"Generation job {job_id} failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        # Update job status to failed
        job = engine.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error_message = error_msg
            job.completed_at = time.time()
        
        raise JobExecutionError(job_id, error_msg)

# API Endpoints

@router.post("/", response_model=JobSubmissionResponse)
async def submit_generation_job(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    engine: SerenityEngine = Depends(get_serenity_engine)
):
    """
    Submit a synthetic data generation job.
    
    This endpoint accepts a generation request and submits it for asynchronous processing.
    Returns a job ID that can be used to track progress and retrieve results.
    """
    try:
        # Validate request
        if request.training_data_path:
            data_path = Path(request.training_data_path)
            if not data_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Training data file not found: {request.training_data_path}"
                )
        
        # Convert priority to JobPriority enum
        priority_map = {1: JobPriority.LOW, 5: JobPriority.NORMAL, 8: JobPriority.HIGH, 10: JobPriority.URGENT}
        job_priority = priority_map.get(request.priority, JobPriority.NORMAL)
        
        # Register the model type if not already registered
        model_name = request.model_type.value
        if model_name not in engine.list_models():
            if model_name == "ctgan":
                from ...models.tabular.ctgan import CTGANGenerator
                # Create a dummy instance for registration to avoid SerenityEngine issues
                dummy_config = CTGANConfig(name=model_name, data_type=DataType.TABULAR)
                dummy_instance = CTGANGenerator(dummy_config)
                engine.register_model(model_name, dummy_instance)
        
        # Create job using SerenityEngine's create_job method
        job = engine.create_job(
            model_name=model_name,
            data_path=request.training_data_path or "",
            n_samples=request.n_samples,
            model_config=request.config or {},
            priority=job_priority
        )
        
        # Store additional request parameters in the job
        if hasattr(job, 'metadata'):
            job.metadata.update({
                'output_format': request.output_format.value,
                'privacy_level': request.privacy_level.value,
                'timeout_minutes': request.timeout_minutes,
                'healthcare_mode': request.healthcare_mode
            })
        
        # Submit job to SerenityEngine
        job_id = engine.submit_job(job)
        
        # Start background task
        background_tasks.add_task(execute_generation_job, job_id, request, engine)
        
        logger.info(f"📝 Generation job submitted: {job_id}")
        
        return JobSubmissionResponse(
            success=True,
            message="Generation job submitted successfully",
            job_id=job_id,
            status=JobStatus.PENDING,
            estimated_duration_minutes=5.0  # TODO: Calculate based on request
        )
        
    except MirageException as e:
        logger.error(f"❌ Mirage error in job submission: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Unexpected error in job submission: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    engine: SerenityEngine = Depends(get_serenity_engine)
):
    """
    Get the status of a generation job.
    
    Returns detailed information about job progress, timing, and results.
    """
    try:
        job_info = engine.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobStatusResponse(
            success=True,
            message="Job status retrieved successfully",
            job_id=job_id,
            status=job_info.get("status", JobStatus.PENDING),
            progress_percentage=job_info.get("progress", 0),
            created_at=job_info.get("created_at"),
            started_at=job_info.get("started_at"),
            completed_at=job_info.get("completed_at"),
            duration_seconds=job_info.get("duration_seconds"),
            current_step=job_info.get("current_step"),
            output_path=job_info.get("output_path"),
            samples_generated=job_info.get("samples_generated", job_info.get("n_samples") if job_info.get("status") == JobStatus.COMPLETED else None),
            error_message=job_info.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving job status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[JobStatus] = None,
    limit: int = 50,
    offset: int = 0,
    engine: SerenityEngine = Depends(get_serenity_engine)
):
    """
    List generation jobs with optional filtering.
    
    Returns a paginated list of jobs with their current status.
    """
    try:
        # Get all jobs from SerenityEngine
        all_jobs = engine.list_jobs()
        
        # Apply status filter if provided
        if status_filter:
            if isinstance(all_jobs[0], dict) if all_jobs else True:
                # Jobs are dicts
                filtered_jobs = [job for job in all_jobs if job.get('status') == status_filter]
            else:
                # Jobs are objects
                filtered_jobs = [job for job in all_jobs if job.status == status_filter]
        else:
            filtered_jobs = all_jobs
        
        # Apply pagination
        total_count = len(filtered_jobs)
        paginated_jobs = filtered_jobs[offset:offset + limit] if limit else filtered_jobs[offset:]
        
        # Convert jobs to response format with proper dict access
        job_responses = []
        for job in paginated_jobs:
            # Handle both dict and object types
            if isinstance(job, dict):
                job_data = job
                job_obj = None
            else:
                job_data = {
                    'job_id': job.job_id,
                    'status': job.status,
                    'created_at': job.created_at,
                    'started_at': getattr(job, 'started_at', None),
                    'completed_at': getattr(job, 'completed_at', None),
                    'progress': getattr(job, 'progress', 0),
                    'current_step': getattr(job, 'current_step', None),
                    'output_path': getattr(job, 'output_path', None),
                    'error_message': getattr(job, 'error_message', None),
                    'n_samples': job.n_samples
                }
                job_obj = job
            
            job_response = JobStatusResponse(
                success=True,
                message="",
                job_id=job_data['job_id'],
                status=job_data['status'],
                progress_percentage=job_data.get('progress', 0),
                created_at=job_data.get('created_at'),
                started_at=job_data.get('started_at'),
                completed_at=job_data.get('completed_at'),
                current_step=job_data.get('current_step'),
                output_path=job_data.get('output_path'),
                samples_generated=job_data.get('n_samples') if job_data.get('status') == JobStatus.COMPLETED else None,
                error_message=job_data.get('error_message')
            )
            job_responses.append(job_response)
        
        return JobListResponse(
            success=True,
            message=f"Retrieved {len(job_responses)} jobs",
            jobs=job_responses,
            total_count=total_count,
            page=offset // limit + 1,
            page_size=limit,
            has_next=offset + limit < total_count
        )
        
    except Exception as e:
        logger.error(f"❌ Error listing jobs: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    engine: SerenityEngine = Depends(get_serenity_engine)
):
    """
    Cancel a pending or running generation job.
    """
    try:
        success = engine.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or cannot be cancelled"
            )
        
        return BaseResponse(
            success=True,
            message=f"Job {job_id} cancelled successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error cancelling job: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get("/jobs/{job_id}/download")
async def download_generated_data(
    job_id: str,
    engine: SerenityEngine = Depends(get_serenity_engine)
):
    """
    Download the generated data from a completed job.
    """
    try:
        job_info = engine.get_job_status(job_id)
        
        if not job_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job_status = job_info.get("status")
        if job_status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} is not completed. Status: {job_status}"
            )
        
        output_path_str = job_info.get("output_path")
        if not output_path_str:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Output file not found for job {job_id}"
            )
        
        output_path = Path(output_path_str)
        if not output_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Output file no longer exists: {output_path}"
            )
        
        return FileResponse(
            path=output_path,
            filename=output_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading file: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/upload", response_model=FileUploadResponse)
async def upload_training_data(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    data_type: DataType = Form(DataType.TABULAR),
    contains_phi: bool = Form(False),
    anonymization_required: bool = Form(True)
):
    """
    Upload training data file for synthetic data generation.
    
    Supports CSV, Excel, and JSON formats with automatic validation.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Check file size
        max_size = settings.max_upload_size_mb * 1024 * 1024
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
            )
        
        # Generate unique file ID
        file_id = f"upload_{uuid.uuid4().hex[:12]}"
        
        # Save file
        file_path = config.storage_paths["datasets"] / f"{file_id}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"📁 File uploaded: {file.filename} -> {file_path}")
        
        return FileUploadResponse(
            success=True,
            message="File uploaded successfully",
            file_id=file_id,
            file_path=str(file_path),
            original_filename=file.filename,
            file_size_bytes=len(content),
            content_type=file.content_type or "application/octet-stream",
            processing_required=True,
            estimated_processing_time_minutes=2.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading file: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/validate-data", response_model=DataValidationResponse)
async def validate_data(request: DataPreviewRequest):
    """
    Validate and preview uploaded data file.
    
    Returns data quality report and healthcare compliance assessment.
    """
    try:
        data_path = Path(request.data_path)
        if not data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data file not found: {request.data_path}"
            )
        
        # Load and analyze data
        loader = HealthcareDataLoader()
        data = loader.load_csv(data_path)
        
        # Get quality report
        quality_report_data = loader.get_data_quality_report()
        quality_report = DataQualityReport(
            total_rows=quality_report_data.total_rows,
            total_columns=quality_report_data.total_columns,
            missing_values_count=sum(quality_report_data.missing_values.values()),
            missing_percentage=(sum(quality_report_data.missing_values.values()) / 
                              (quality_report_data.total_rows * quality_report_data.total_columns)) * 100,
            quality_score=quality_report_data.quality_score,
            column_stats={},  # TODO: Calculate detailed stats
            data_types=quality_report_data.data_types,
            issues=quality_report_data.issues,
            recommendations=quality_report_data.recommendations
        )
        
        # Get compliance report
        compliance_data = loader.validate_healthcare_compliance(data)
        compliance_report = HealthcareComplianceReport(
            hipaa_compliant=compliance_data["hipaa_compliant"],
            risk_level=compliance_data["risk_level"],
            potential_phi_columns=[],  # TODO: Extract from compliance data
            phi_risk_score=0.1 if compliance_data["hipaa_compliant"] else 0.8,
            compliance_issues=compliance_data["issues"],
            recommendations=compliance_data["recommendations"],
            anonymization_applied=False,
            anonymization_method=None
        )
        
        return DataValidationResponse(
            success=True,
            message="Data validation completed",
            data_path=str(data_path),
            quality_report=quality_report,
            compliance_report=compliance_report,
            is_valid=quality_report.quality_score > 0.7 and compliance_report.hipaa_compliant,
            validation_errors=[],
            warnings=quality_report.issues
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error validating data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/validate-config", response_model=ConfigValidationResponse)
async def validate_config(request: ConfigValidationRequest):
    """
    Validate model configuration parameters.
    
    Ensures configuration is valid for the specified model type.
    """
    try:
        if request.model_type == ModelType.CTGAN:
            # Validate CTGAN configuration with required fields
            config_dict = request.config.copy()
            config_dict.update({
                'name': config_dict.get('name', 'ctgan'),
                'data_type': DataType.TABULAR
            })
            
            try:
                validated_config = CTGANConfig(**config_dict)
                return ConfigValidationResponse(
                    success=True,
                    message="Configuration is valid",
                    is_valid=True,
                    validated_config=validated_config.model_dump()
                )
            except Exception as e:
                return ConfigValidationResponse(
                    success=True,
                    message="Configuration validation completed with errors",
                    is_valid=False,
                    validation_errors=[str(e)]
                )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation not implemented for model type: {request.model_type}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error validating config: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/batch", response_model=BatchJobResponse)
async def submit_batch_jobs(
    request: BatchGenerateRequest,
    background_tasks: BackgroundTasks,
    engine: SerenityEngine = Depends(get_serenity_engine)
):
    """
    Submit multiple generation jobs as a batch.
    
    All jobs in the batch can be tracked together and run sequentially or in parallel.
    """
    try:
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        job_ids = []
        
        for i, job_request in enumerate(request.jobs):
            # Convert priority to JobPriority enum
            priority_map = {1: JobPriority.LOW, 5: JobPriority.NORMAL, 8: JobPriority.HIGH, 10: JobPriority.URGENT}
            job_priority = priority_map.get(job_request.priority, JobPriority.NORMAL)
            
            # Register the model type if not already registered
            model_name = job_request.model_type.value
            if model_name not in engine.list_models():
                if model_name == "ctgan":
                    from ...models.tabular.ctgan import CTGANGenerator

                    dummy_config = CTGANConfig(name=model_name, data_type=DataType.TABULAR)
                    dummy_instance = CTGANGenerator(dummy_config)
                    engine.register_model(model_name, dummy_instance)
            
            # Create job using SerenityEngine's create_job method
            job = engine.create_job(
                model_name=model_name,
                data_path=job_request.training_data_path or "",
                n_samples=job_request.n_samples,
                model_config=job_request.config or {},
                priority=job_priority
            )
            
            # Store additional request parameters in the job
            if hasattr(job, 'metadata'):
                job.metadata.update({
                    'output_format': job_request.output_format.value,
                    'privacy_level': job_request.privacy_level.value,
                    'timeout_minutes': job_request.timeout_minutes,
                    'healthcare_mode': getattr(job_request, 'healthcare_mode', True)
                })
            
            # Submit job to SerenityEngine
            job_id = engine.submit_job(job)
            job_ids.append(job_id)
            
            # Add background task
            background_tasks.add_task(execute_generation_job, job_id, job_request, engine)
        
        logger.info(f"📦 Batch job submitted: {batch_id} with {len(job_ids)} jobs")
        
        return BatchJobResponse(
            success=True,
            message=f"Batch job submitted with {len(job_ids)} individual jobs",
            batch_id=batch_id,
            job_ids=job_ids,
            total_jobs=len(job_ids),
            completed_jobs=0,
            failed_jobs=0,
            overall_status=JobStatus.PENDING
        )
        
    except Exception as e:
        logger.error(f"❌ Error submitting batch jobs: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")