"""
SerenityEngine - Core Generation Orchestrator for Mirage ML Engine
Serenity named after Saanya meaning serenity, this is the heart of synthetic data generation.

The SerenityEngine coordinates all ML model execution, resource management,
and pipeline orchestration with the calm reliability as its name suggests.
"""

import asyncio
import time
import uuid
import psutil
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
import pandas as pd
import numpy as np
from loguru import logger

from ..core.config import config, settings, device
from ..core.base import BaseGenerator, TrainingProgress, ModelStatus
from ..core.exceptions import (
    JobError,
    JobExecutionError,
    JobTimeoutError,
    ResourceError,
    ModelError,
    DataError
)


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class GenerationJob:
    """Represents a synthetic data generation job."""
    
    # Job identification
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Job configuration
    model_name: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    data_path: Optional[str] = None
    n_samples: int = 1000
    
    # Job metadata
    priority: JobPriority = JobPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Job state
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    
    # Results
    output_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Resource tracking
    estimated_memory_mb: Optional[float] = None
    actual_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if job is currently active."""
        return self.status in [JobStatus.QUEUED, JobStatus.RUNNING]
    
    @property
    def is_finished(self) -> bool:
        """Check if job is finished (success or failure)."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]


class ResourceMonitor:
    """Monitors system resources during generation."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": [],
            "mps_available": config.system.mps_available
        }
        self._monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": [],
            "mps_available": config.system.mps_available
        }
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.debug("🔍 Resource monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        # Calculate summary statistics
        summary = {}
        for metric, values in self.metrics.items():
            if metric == "mps_available":
                summary[metric] = values
            elif values and isinstance(values, list) and len(values) > 0:
                # Check if all values are numeric
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    summary[f"{metric}_avg"] = np.mean(numeric_values)
                    summary[f"{metric}_max"] = np.max(numeric_values)
                    summary[f"{metric}_min"] = np.min(numeric_values)
                else:
                    summary[metric] = values
            else:
                summary[metric] = values
        
        logger.debug("🔍 Resource monitoring stopped")
        return summary
    
    def _monitor_loop(self, interval: float):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory metrics
                self.metrics["cpu_percent"].append(psutil.cpu_percent(interval=0.1))
                memory = psutil.virtual_memory()
                self.metrics["memory_percent"].append(memory.percent)
                self.metrics["memory_mb"].append(memory.used / 1024 / 1024)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": memory.percent,
                "memory_mb": memory.used / 1024 / 1024,
                "available_memory_mb": memory.available / 1024 / 1024
            }
        except Exception:
            return {}


class SerenityEngine:
    """
    SerenityEngine - The heart of Mirage's synthetic data generation.
    
    Named after Serenity, this engine provides calm, reliable coordination
    of all ML model execution, resource management, and generation workflows.
    """
    
    def __init__(self, max_concurrent_jobs: int = None):
        """
        Initialize SerenityEngine.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent generation jobs
        """
        self.max_concurrent_jobs = max_concurrent_jobs or settings.max_concurrent_jobs
        
        # Job management
        self.jobs: Dict[str, GenerationJob] = {}
        self.active_jobs: Dict[str, GenerationJob] = {}
        self._job_lock = threading.Lock()
        
        # Resource management
        self.resource_monitor = ResourceMonitor()
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_jobs,
            thread_name_prefix="SerenityEngine"
        )
        
        # Model registry
        self.models: Dict[str, BaseGenerator] = {}
        
        # Callbacks
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info(f"💙 SerenityEngine initialized with {self.max_concurrent_jobs} max concurrent jobs")
        logger.info(f"🖥️  Device: {device}")
        logger.info(f"🧠 Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    
    def register_model(self, name: str, model: BaseGenerator) -> None:
        """
        Register a model with SerenityEngine.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
        logger.info(f"📝 Model '{name}' registered with SerenityEngine")
    
    def get_model(self, name: str) -> Optional[BaseGenerator]:
        """Get a registered model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())
    
    def create_job(
        self,
        model_name: str,
        data_path: str,
        n_samples: int,
        model_config: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        **kwargs
    ) -> GenerationJob:
        """
        Create a new generation job.
        
        Args:
            model_name: Name of the model to use
            data_path: Path to training data
            n_samples: Number of samples to generate
            model_config: Model configuration parameters
            priority: Job priority level
            **kwargs: Additional job parameters
            
        Returns:
            Created GenerationJob
            
        Raises:
            ModelError: If model is not registered
        """
        if model_name not in self.models:
            raise ModelError(f"Model '{model_name}' not registered with SerenityEngine")
        
        job = GenerationJob(
            model_name=model_name,
            model_config=model_config or {},
            data_path=data_path,
            n_samples=n_samples,
            priority=priority,
            **kwargs
        )
        
        with self._job_lock:
            self.jobs[job.job_id] = job
        
        logger.info(f"📋 Job {job.job_id} created for model '{model_name}'")
        return job
    
    def submit_job(self, job: GenerationJob) -> str:
        """
        Submit a job for execution.
        
        Args:
            job: Job to submit
            
        Returns:
            Job ID
            
        Raises:
            ResourceError: If no resources available
        """
        # Check resource availability
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            raise ResourceError(
                f"Maximum concurrent jobs ({self.max_concurrent_jobs}) reached",
                resource_type="job_slots",
                available=self.max_concurrent_jobs - len(self.active_jobs),
                required=1
            )
        
        # Check memory availability
        current_usage = self.resource_monitor.get_current_usage()
        if current_usage.get("memory_percent", 0) > 90:
            raise ResourceError(
                "Insufficient memory for new job",
                resource_type="memory",
                available=current_usage.get("available_memory_mb", 0),
                required=job.estimated_memory_mb or 1000
            )
        
        # Update job status and submit
        job.status = JobStatus.QUEUED
        
        with self._job_lock:
            self.active_jobs[job.job_id] = job
        
        # Submit to thread pool
        future = self.executor.submit(self._execute_job, job)
        
        logger.info(f"🚀 Job {job.job_id} submitted to SerenityEngine")
        return job.job_id
    
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancelled, False if not found or already finished
        """
        job = self.jobs.get(job_id)
        if not job or job.is_finished:
            return False
        
        job.status = JobStatus.CANCELLED
        job.completed_at = time.time()
        
        with self._job_lock:
            self.active_jobs.pop(job_id, None)
        
        logger.info(f"❌ Job {job_id} cancelled")
        return True
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary
        """
        job = self.jobs.get(job_id)
        if not job:
            raise JobError(f"Job {job_id} not found")
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "duration_seconds": job.duration_seconds,
            "model_name": job.model_name,
            "n_samples": job.n_samples,
            "output_path": job.output_path,
            "error_message": job.error_message,
            "metrics": job.metrics
        }
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
        """
        List jobs with optional status filter.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of job status dictionaries
        """
        jobs = []
        for job in self.jobs.values():
            if status is None or job.status == status:
                jobs.append(self.get_job_status(job.job_id))
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        return jobs
    
    def add_progress_callback(self, job_id: str, callback: Callable[[float, str], None]) -> None:
        """
        Add a progress callback for a job.
        
        Args:
            job_id: Job ID
            callback: Callback function that takes (progress, message)
        """
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)
    
    def _execute_job(self, job: GenerationJob) -> None:
        """
        Execute a generation job.
        
        This is the core generation logic that runs in a thread pool.
        """
        job.started_at = time.time()
        job.status = JobStatus.RUNNING
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            logger.info(f"💙 SerenityEngine starting job {job.job_id}")
            
            # Get the model
            model = self.models[job.model_name]
            if not model:
                raise ModelError(f"Model '{job.model_name}' not found. Available models: {list(self.models.keys())}")
            
            # Load and validate data
            self._update_job_progress(job, 0.1, "Loading data...")
            data = self._load_data(job.data_path)
            
            # Estimate memory usage
            job.estimated_memory_mb = model.estimate_memory_usage(job.n_samples)
            
            # Train the model
            self._update_job_progress(job, 0.2, "Training model...")
            model.fit(data, **job.model_config)
            
            # Generate synthetic data
            self._update_job_progress(job, 0.8, "Generating synthetic data...")
            generated_data = model.generate(job.n_samples)
            
            # Save results
            self._update_job_progress(job, 0.95, "Saving results...")
            output_path = self._save_results(job, generated_data)
            job.output_path = output_path
            
            # Stop resource monitoring and collect metrics
            resource_metrics = self.resource_monitor.stop_monitoring()
            job.metrics.update(resource_metrics)
            
            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.progress = 1.0
            
            logger.info(f"✅ SerenityEngine completed job {job.job_id} in {job.duration_seconds:.2f}s")
            
        except Exception as e:
            # Handle job failure
            job.status = JobStatus.FAILED
            job.completed_at = time.time()
            job.error_message = str(e)
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            logger.error(f"❌ SerenityEngine job {job.job_id} failed: {e}")
            
        finally:
            # Remove from active jobs
            with self._job_lock:
                self.active_jobs.pop(job.job_id, None)
    
    def _update_job_progress(self, job: GenerationJob, progress: float, message: str) -> None:
        """Update job progress and notify callbacks."""
        job.progress = progress
        
        # Call progress callbacks
        callbacks = self.progress_callbacks.get(job.job_id, [])
        for callback in callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        logger.debug(f"🔄 Job {job.job_id}: {progress*100:.1f}% - {message}")
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from path."""
        try:
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path)
            else:
                raise DataError(f"Unsupported data format: {data_path}")
        except Exception as e:
            raise DataError(f"Failed to load data from {data_path}: {e}")
    
    def _save_results(self, job: GenerationJob, data: pd.DataFrame) -> str:
        """Save generation results."""
        output_dir = config.storage_paths["outputs"] / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "synthetic_data.csv"
        data.to_csv(output_path, index=False)
        
        # Save job metadata
        metadata_path = output_dir / "job_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                "job_id": job.job_id,
                "model_name": job.model_name,
                "n_samples": job.n_samples,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "duration_seconds": job.duration_seconds
            }, f, indent=2)
        
        return str(output_path)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        total_jobs = len(self.jobs)
        active_jobs = len(self.active_jobs)
        completed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED])
        failed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])
        
        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": completed_jobs / max(total_jobs, 1) * 100,
            "registered_models": len(self.models),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "device": str(device),
            "mps_available": config.system.mps_available
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown SerenityEngine."""
        logger.info("🛑 SerenityEngine shutting down...")
        
        # Cancel all active jobs
        with self._job_lock:
            for job_id in list(self.active_jobs.keys()):
                self.cancel_job(job_id)
        
        # Shutdown thread pool (compatible with Python 3.8+)
        self.executor.shutdown(wait=True)
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        logger.info("✅ SerenityEngine shutdown complete")


# Global SerenityEngine instance
_serenity_engine: Optional[SerenityEngine] = None


def get_serenity_engine() -> SerenityEngine:
    """Get the global SerenityEngine instance."""
    global _serenity_engine
    if _serenity_engine is None:
        _serenity_engine = SerenityEngine()
    return _serenity_engine


def shutdown_serenity_engine() -> None:
    """Shutdown the global SerenityEngine instance."""
    global _serenity_engine
    if _serenity_engine is not None:
        _serenity_engine.shutdown()
        _serenity_engine = None


# Export all public classes and functions
__all__ = [
    "SerenityEngine",
    "GenerationJob", 
    "JobStatus",
    "JobPriority",
    "ResourceMonitor",
    "get_serenity_engine",
    "shutdown_serenity_engine"
]