"""
Job Queue System for Mirage ML Engine
Provides persistent job storage, priority queuing, and scheduling for SerenityEngine.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import asdict
from collections import defaultdict, deque
from enum import Enum
import heapq

from loguru import logger

from ..core.config import config
from ..core.exceptions import JobError, JobNotFoundError
from .serenity_engine import GenerationJob, JobStatus, JobPriority


class QueueStatus(str, Enum):
    """Queue status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


class JobQueue:
    """
    Advanced job queue with priority handling, persistence, and scheduling.
    Works seamlessly with SerenityEngine for robust job management.
    """
    
    def __init__(self, queue_name: str = "default", persistent: bool = True):
        """
        Initialize job queue.
        
        Args:
            queue_name: Name of the queue
            persistent: Whether to persist jobs to disk
        """
        self.queue_name = queue_name
        self.persistent = persistent
        self.status = QueueStatus.ACTIVE
        
        # Thread-safe queue operations
        self._lock = threading.RLock()
        
        # Priority queue (heap) for pending jobs
        # Format: (priority_value, created_at, job_id)
        self._priority_queue: List[tuple] = []
        
        # Job storage by status
        self._jobs: Dict[str, GenerationJob] = {}
        self._jobs_by_status: Dict[JobStatus, set] = defaultdict(set)
        
        # Queue statistics
        self._stats = {
            "total_enqueued": 0,
            "total_dequeued": 0,
            "total_completed": 0,
            "total_failed": 0,
            "queue_created_at": time.time()
        }
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Persistence setup
        if self.persistent:
            self._setup_persistence()
            self._load_persisted_jobs()
        
        logger.info(f"📋 JobQueue '{queue_name}' initialized (persistent: {persistent})")
    
    def _setup_persistence(self) -> None:
        """Setup persistent storage directories."""
        self.storage_dir = config.storage_paths["base"] / "queue" / self.queue_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs_file = self.storage_dir / "jobs.json"
        self.stats_file = self.storage_dir / "stats.json"
        
        logger.debug(f"📁 Queue persistence: {self.storage_dir}")
    
    def _load_persisted_jobs(self) -> None:
        """Load jobs from persistent storage."""
        if not self.jobs_file.exists():
            return
        
        try:
            with open(self.jobs_file, 'r') as f:
                data = json.load(f)
            
            # Restore jobs
            for job_data in data.get("jobs", []):
                job = self._deserialize_job(job_data)
                self._add_job_to_internal_structures(job)
            
            # Restore stats
            self._stats.update(data.get("stats", {}))
            
            logger.info(f"📥 Restored {len(self._jobs)} jobs from persistence")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load persisted jobs: {e}")
    
    def _persist_state(self) -> None:
        """Persist current queue state to disk."""
        if not self.persistent:
            return
        
        try:
            data = {
                "queue_name": self.queue_name,
                "saved_at": time.time(),
                "jobs": [self._serialize_job(job) for job in self._jobs.values()],
                "stats": self._stats
            }
            
            # Atomic write
            temp_file = self.jobs_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            temp_file.replace(self.jobs_file)
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to persist queue state: {e}")
    
    def _serialize_job(self, job: GenerationJob) -> Dict[str, Any]:
        """Serialize job to JSON-compatible format."""
        return {
            **asdict(job),
            "status": job.status.value,
            "priority": job.priority.value
        }
    
    def _deserialize_job(self, data: Dict[str, Any]) -> GenerationJob:
        """Deserialize job from JSON data."""
        # Convert enum values back
        data["status"] = JobStatus(data["status"])
        data["priority"] = JobPriority(data["priority"])
        
        return GenerationJob(**data)
    
    def _add_job_to_internal_structures(self, job: GenerationJob) -> None:
        """Add job to internal data structures."""
        self._jobs[job.job_id] = job
        self._jobs_by_status[job.status].add(job.job_id)
        
        # Add to priority queue if pending
        if job.status == JobStatus.PENDING:
            priority_value = self._get_priority_value(job.priority)
            heapq.heappush(self._priority_queue, (priority_value, job.created_at, job.job_id))
    
    def _get_priority_value(self, priority: JobPriority) -> int:
        """Convert priority enum to numeric value for heap (lower = higher priority)."""
        priority_map = {
            JobPriority.URGENT: 1,
            JobPriority.HIGH: 2,
            JobPriority.NORMAL: 3,
            JobPriority.LOW: 4
        }
        return priority_map.get(priority, 3)
    
    def enqueue(self, job: GenerationJob) -> None:
        """
        Add job to the queue.
        
        Args:
            job: Job to enqueue
        """
        if self.status != QueueStatus.ACTIVE:
            raise JobError(f"Queue '{self.queue_name}' is not active (status: {self.status})")
        
        with self._lock:
            # Update job status
            if job.status == JobStatus.PENDING:
                # Add to priority queue
                priority_value = self._get_priority_value(job.priority)
                heapq.heappush(self._priority_queue, (priority_value, job.created_at, job.job_id))
            
            # Add to internal structures
            self._add_job_to_internal_structures(job)
            
            # Update stats
            self._stats["total_enqueued"] += 1
            
            # Persist state
            self._persist_state()
            
            # Trigger callbacks
            self._trigger_callbacks("job_enqueued", job)
        
        logger.info(f"📥 Job {job.job_id} enqueued to '{self.queue_name}' (priority: {job.priority.value})")
    
    def dequeue(self, count: int = 1) -> List[GenerationJob]:
        """
        Dequeue jobs from the queue based on priority.
        
        Args:
            count: Number of jobs to dequeue
            
        Returns:
            List of dequeued jobs
        """
        if self.status != QueueStatus.ACTIVE:
            return []
        
        dequeued_jobs = []
        
        with self._lock:
            for _ in range(count):
                if not self._priority_queue:
                    break
                
                # Get highest priority job
                priority_value, created_at, job_id = heapq.heappop(self._priority_queue)
                
                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.PENDING:
                    # Update job status
                    job.status = JobStatus.QUEUED
                    self._jobs_by_status[JobStatus.PENDING].discard(job_id)
                    self._jobs_by_status[JobStatus.QUEUED].add(job_id)
                    
                    dequeued_jobs.append(job)
                    self._stats["total_dequeued"] += 1
                    
                    # Trigger callbacks
                    self._trigger_callbacks("job_dequeued", job)
            
            # Persist state if any jobs were dequeued
            if dequeued_jobs:
                self._persist_state()
        
        if dequeued_jobs:
            job_ids = [job.job_id for job in dequeued_jobs]
            logger.info(f"📤 Dequeued {len(dequeued_jobs)} jobs from '{self.queue_name}': {job_ids}")
        
        return dequeued_jobs
    
    def update_job_status(self, job_id: str, status: JobStatus, error_message: Optional[str] = None) -> bool:
        """
        Update job status in the queue.
        
        Args:
            job_id: Job ID to update
            status: New status
            error_message: Optional error message for failed jobs
            
        Returns:
            True if updated, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            # Update status tracking
            old_status = job.status
            self._jobs_by_status[old_status].discard(job_id)
            self._jobs_by_status[status].add(job_id)
            
            # Update job
            job.status = status
            if error_message:
                job.error_message = error_message
            
            if status == JobStatus.COMPLETED:
                job.completed_at = time.time()
                self._stats["total_completed"] += 1
            elif status == JobStatus.FAILED:
                job.completed_at = time.time()
                self._stats["total_failed"] += 1
            
            # Persist state
            self._persist_state()
            
            # Trigger callbacks
            self._trigger_callbacks("job_status_updated", job)
        
        logger.info(f"🔄 Job {job_id} status updated: {old_status.value} → {status.value}")
        return True
    
    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(
        self, 
        status: Optional[JobStatus] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[GenerationJob]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            
        Returns:
            List of jobs
        """
        with self._lock:
            if status:
                job_ids = list(self._jobs_by_status[status])
            else:
                job_ids = list(self._jobs.keys())
            
            # Sort by creation time (newest first)
            job_ids.sort(key=lambda jid: self._jobs[jid].created_at, reverse=True)
            
            # Apply pagination
            if limit:
                job_ids = job_ids[offset:offset + limit]
            else:
                job_ids = job_ids[offset:]
            
            return [self._jobs[job_id] for job_id in job_ids]
    
    def get_pending_count(self) -> int:
        """Get number of pending jobs."""
        return len(self._jobs_by_status[JobStatus.PENDING])
    
    def get_queue_size(self) -> int:
        """Get total number of jobs in queue."""
        return len(self._jobs)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        with self._lock:
            status_counts = {
                status.value: len(job_ids) 
                for status, job_ids in self._jobs_by_status.items()
            }
            
            # Calculate average wait time for completed jobs
            completed_jobs = [
                self._jobs[job_id] 
                for job_id in self._jobs_by_status[JobStatus.COMPLETED]
                if self._jobs[job_id].duration_seconds is not None
            ]
            
            avg_processing_time = 0.0
            if completed_jobs:
                avg_processing_time = sum(
                    job.duration_seconds for job in completed_jobs
                ) / len(completed_jobs)
            
            return {
                "queue_name": self.queue_name,
                "status": self.status.value,
                "total_jobs": len(self._jobs),
                "pending_jobs": len(self._priority_queue),
                "status_breakdown": status_counts,
                "average_processing_time": avg_processing_time,
                "queue_uptime": time.time() - self._stats["queue_created_at"],
                **self._stats
            }
    
    def clear_completed_jobs(self, older_than_hours: Optional[float] = None) -> int:
        """
        Clear completed jobs from the queue.
        
        Args:
            older_than_hours: Only clear jobs completed more than this many hours ago
            
        Returns:
            Number of jobs cleared
        """
        cleared_count = 0
        cutoff_time = None
        
        if older_than_hours:
            cutoff_time = time.time() - (older_than_hours * 3600)
        
        with self._lock:
            jobs_to_remove = []
            
            for job_id in self._jobs_by_status[JobStatus.COMPLETED]:
                job = self._jobs[job_id]
                
                if cutoff_time is None or (job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            # Remove jobs
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                self._jobs_by_status[JobStatus.COMPLETED].discard(job_id)
                cleared_count += 1
            
            # Persist state
            if cleared_count > 0:
                self._persist_state()
        
        if cleared_count > 0:
            logger.info(f"🧹 Cleared {cleared_count} completed jobs from queue '{self.queue_name}'")
        
        return cleared_count
    
    def pause_queue(self) -> None:
        """Pause the queue (stop processing new jobs)."""
        with self._lock:
            self.status = QueueStatus.PAUSED
            self._persist_state()
        
        logger.info(f"⏸️  Queue '{self.queue_name}' paused")
    
    def resume_queue(self) -> None:
        """Resume the queue."""
        with self._lock:
            self.status = QueueStatus.ACTIVE
            self._persist_state()
        
        logger.info(f"▶️  Queue '{self.queue_name}' resumed")
    
    def add_callback(self, event: str, callback: Callable[[GenerationJob], None]) -> None:
        """
        Add callback for queue events.
        
        Args:
            event: Event name (job_enqueued, job_dequeued, job_status_updated)
            callback: Callback function
        """
        self._callbacks[event].append(callback)
        logger.debug(f"📞 Callback added for event '{event}' in queue '{self.queue_name}'")
    
    def _trigger_callbacks(self, event: str, job: GenerationJob) -> None:
        """Trigger callbacks for an event."""
        for callback in self._callbacks[event]:
            try:
                callback(job)
            except Exception as e:
                logger.warning(f"⚠️  Callback error for event '{event}': {e}")


class QueueManager:
    """
    Manages multiple job queues with different priorities and purposes.
    Provides a centralized interface for queue operations.
    """
    
    def __init__(self):
        """Initialize queue manager."""
        self.queues: Dict[str, JobQueue] = {}
        self._lock = threading.RLock()
        
        # Create default queues
        self.create_queue("high_priority", persistent=True)
        self.create_queue("normal", persistent=True)
        self.create_queue("low_priority", persistent=True)
        
        logger.info("🗂️  QueueManager initialized with default queues")
    
    def create_queue(self, name: str, persistent: bool = True) -> JobQueue:
        """
        Create a new queue.
        
        Args:
            name: Queue name
            persistent: Whether to persist jobs
            
        Returns:
            Created queue
        """
        with self._lock:
            if name in self.queues:
                return self.queues[name]
            
            queue = JobQueue(name, persistent=persistent)
            self.queues[name] = queue
            
            logger.info(f"📋 Queue '{name}' created")
            return queue
    
    def get_queue(self, name: str) -> Optional[JobQueue]:
        """Get queue by name."""
        return self.queues.get(name)
    
    def enqueue_job(self, job: GenerationJob, queue_name: Optional[str] = None) -> None:
        """
        Enqueue job to appropriate queue based on priority.
        
        Args:
            job: Job to enqueue
            queue_name: Specific queue name (auto-selected if None)
        """
        if queue_name is None:
            # Auto-select queue based on priority
            queue_name = {
                JobPriority.URGENT: "high_priority",
                JobPriority.HIGH: "high_priority",
                JobPriority.NORMAL: "normal",
                JobPriority.LOW: "low_priority"
            }.get(job.priority, "normal")
        
        queue = self.get_queue(queue_name)
        if not queue:
            raise JobError(f"Queue '{queue_name}' not found")
        
        queue.enqueue(job)
    
    def get_next_jobs(self, count: int = 1) -> List[GenerationJob]:
        """
        Get next jobs from queues in priority order.
        
        Args:
            count: Number of jobs to get
            
        Returns:
            List of jobs in priority order
        """
        jobs = []
        remaining = count
        
        # Check queues in priority order
        queue_priority = ["high_priority", "normal", "low_priority"]
        
        for queue_name in queue_priority:
            if remaining <= 0:
                break
            
            queue = self.get_queue(queue_name)
            if queue:
                queue_jobs = queue.dequeue(remaining)
                jobs.extend(queue_jobs)
                remaining -= len(queue_jobs)
        
        return jobs
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues."""
        with self._lock:
            return {
                name: queue.get_queue_stats()
                for name, queue in self.queues.items()
            }
    
    def cleanup_all_queues(self, older_than_hours: float = 24) -> Dict[str, int]:
        """
        Cleanup completed jobs from all queues.
        
        Args:
            older_than_hours: Clear jobs older than this
            
        Returns:
            Dictionary of queue names and cleared counts
        """
        results = {}
        
        with self._lock:
            for name, queue in self.queues.items():
                cleared = queue.clear_completed_jobs(older_than_hours)
                results[name] = cleared
        
        total_cleared = sum(results.values())
        if total_cleared > 0:
            logger.info(f"🧹 Cleaned up {total_cleared} jobs across all queues")
        
        return results


# Global queue manager instance
_queue_manager: Optional[QueueManager] = None


def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager


def get_queue(name: str) -> Optional[JobQueue]:
    """Get a specific queue by name."""
    return get_queue_manager().get_queue(name)


# Export all public classes and functions
__all__ = [
    "JobQueue",
    "QueueManager", 
    "QueueStatus",
    "get_queue_manager",
    "get_queue"
]