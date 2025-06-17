#!/usr/bin/env python3
"""
Test script for SerenityEngine.
Run this to verify the core generation orchestrator works correctly.
"""

import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class MockCTGANGenerator:
    """Mock CTGAN generator for testing SerenityEngine."""
    
    def __init__(self):
        from mirage.core.base import BaseGenerator, BaseGeneratorConfig, DataType
        
        # Create a test configuration
        config = BaseGeneratorConfig(
            name="mock_ctgan",
            version="1.0.0",
            description="Mock CTGAN for SerenityEngine testing",
            data_type=DataType.TABULAR,
            batch_size=32,
            random_seed=42
        )
        
        # Create concrete implementation
        class ConcreteCTGAN(BaseGenerator):
            def get_config_schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "epochs": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "batch_size": {"type": "integer", "minimum": 16, "maximum": 512}
                    },
                    "required": ["epochs"]
                }
            
            def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
                validated = config.copy()
                validated.setdefault("epochs", 100)
                validated.setdefault("batch_size", 32)
                return validated
            
            def fit(self, data, validation_data=None, **kwargs):
                from mirage.core.base import ModelStatus, TrainingProgress
                
                self._set_status(ModelStatus.TRAINING)
                self.validate_data(data)
                
                if isinstance(data, pd.DataFrame):
                    self._training_data_shape = data.shape
                    self._column_names = list(data.columns)
                
                # Simulate training with realistic timing
                epochs = kwargs.get('epochs', 10)
                for epoch in range(1, epochs + 1):
                    progress = TrainingProgress(
                        epoch=epoch,
                        total_epochs=epochs,
                        loss=1.0 / epoch,
                        elapsed_time=epoch * 0.1
                    )
                    self._update_training_progress(progress)
                    time.sleep(0.05)  # Simulate training time
                
                self.metadata.training_epochs = epochs
                self.metadata.training_samples = len(data)
                self._set_status(ModelStatus.TRAINED)
            
            def generate(self, n_samples: int, **kwargs):
                from mirage.core.base import ModelStatus
                
                if not self.is_trained:
                    raise RuntimeError("Model must be trained before generating")
                
                self._set_status(ModelStatus.GENERATING)
                
                # Generate realistic synthetic data
                if self._column_names:
                    generated_data = pd.DataFrame({
                        col: np.random.randn(n_samples) for col in self._column_names
                    })
                else:
                    shape = (n_samples,) + self._training_data_shape[1:]
                    generated_data = np.random.randn(*shape)
                
                # Simulate generation time
                time.sleep(0.1)
                
                self._set_status(ModelStatus.TRAINED)
                return generated_data
            
            def _save_model_data(self, path: Path) -> None:
                import torch
                model_file = path / "mock_ctgan.pt"
                torch.save({"model": "mock_ctgan"}, model_file)
            
            def _load_model_data(self, path: Path) -> None:
                import torch
                model_file = path / "mock_ctgan.pt"
                if model_file.exists():
                    torch.load(model_file)
        
        self.generator = ConcreteCTGAN(config)
    
    def __getattr__(self, name):
        """Delegate to the generator."""
        return getattr(self.generator, name)


def create_sample_healthcare_data() -> str:
    """Create sample healthcare data for testing."""
    np.random.seed(42)
    
    # Generate realistic healthcare data
    n_patients = 200
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(45, 15, n_patients).astype(int).clip(18, 90),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'bmi': np.random.normal(25, 4, n_patients).round(1).clip(15, 45),
        'blood_pressure_systolic': np.random.normal(120, 20, n_patients).astype(int).clip(90, 200),
        'blood_pressure_diastolic': np.random.normal(80, 15, n_patients).astype(int).clip(60, 120),
        'cholesterol': np.random.normal(200, 40, n_patients).astype(int).clip(120, 400),
        'diabetes': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'heart_disease': np.random.choice([0, 1], n_patients, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name


def test_serenity_engine():
    """Test the SerenityEngine functionality."""
    print("💙 Testing SerenityEngine - The Heart of Mirage")
    print("=" * 60)
    
    try:
        # Import SerenityEngine
        from mirage.engine.serenity_engine import (
            SerenityEngine,
            GenerationJob,
            JobStatus,
            JobPriority,
            ResourceMonitor
        )
        
        print("✅ SerenityEngine imported successfully")
        
        # Test 1: Engine Initialization
        print("\n🧪 Test 1: SerenityEngine Initialization")
        try:
            engine = SerenityEngine(max_concurrent_jobs=2)
            
            print(f"   ✅ SerenityEngine created")
            print(f"   ✅ Max concurrent jobs: {engine.max_concurrent_jobs}")
            print(f"   ✅ Device: {engine.resource_monitor.metrics['mps_available']}")
            
            stats = engine.get_engine_stats()
            print(f"   ✅ Initial stats: {stats['total_jobs']} jobs, {stats['registered_models']} models")
            
        except Exception as e:
            print(f"   ❌ Engine initialization failed: {e}")
            return False
        
        # Test 2: Model Registration
        print("\n🧪 Test 2: Model Registration")
        try:
            # Create and register mock CTGAN
            mock_ctgan = MockCTGANGenerator()
            engine.register_model("mock_ctgan", mock_ctgan)
            
            print(f"   ✅ Model registered: mock_ctgan")
            print(f"   ✅ Available models: {engine.list_models()}")
            
            retrieved_model = engine.get_model("mock_ctgan")
            print(f"   ✅ Model retrieved: {retrieved_model is not None}")
            
        except Exception as e:
            print(f"   ❌ Model registration failed: {e}")
            return False
        
        # Test 3: Resource Monitor
        print("\n🧪 Test 3: Resource Monitor")
        try:
            monitor = ResourceMonitor()
            
            # Test current usage
            usage = monitor.get_current_usage()
            print(f"   ✅ Current CPU: {usage.get('cpu_percent', 0):.1f}%")
            print(f"   ✅ Current Memory: {usage.get('memory_percent', 0):.1f}%")
            print(f"   ✅ Available Memory: {usage.get('available_memory_mb', 0):.0f} MB")
            
            # Test monitoring
            monitor.start_monitoring(interval=0.1)
            time.sleep(0.5)
            summary = monitor.stop_monitoring()
            
            print(f"   ✅ Monitoring collected {len(summary)} metrics")
            if 'cpu_percent_avg' in summary:
                print(f"   ✅ Avg CPU: {summary['cpu_percent_avg']:.1f}%")
            if 'memory_percent_avg' in summary:
                print(f"   ✅ Avg Memory: {summary['memory_percent_avg']:.1f}%")
            print(f"   ✅ MPS Available: {summary.get('mps_available', False)}")
            
        except Exception as e:
            print(f"   ❌ Resource monitor test failed: {e}")
            return False
        
        # Test 4: Job Creation
        print("\n🧪 Test 4: Job Creation")
        try:
            # Create sample data
            data_path = create_sample_healthcare_data()
            print(f"   ✅ Sample healthcare data created: {data_path}")
            
            # Create a job
            job = engine.create_job(
                model_name="mock_ctgan",
                data_path=data_path,
                n_samples=50,
                model_config={"epochs": 5, "batch_size": 32},
                priority=JobPriority.NORMAL
            )
            
            print(f"   ✅ Job created: {job.job_id}")
            print(f"   ✅ Job status: {job.status}")
            print(f"   ✅ Job priority: {job.priority}")
            print(f"   ✅ Model name: {job.model_name}")
            print(f"   ✅ Samples requested: {job.n_samples}")
            
        except Exception as e:
            print(f"   ❌ Job creation failed: {e}")
            return False
        
        # Test 5: Job Submission and Execution
        print("\n🧪 Test 5: Job Submission and Execution")
        try:
            # Submit the job
            job_id = engine.submit_job(job)
            print(f"   ✅ Job submitted: {job_id}")
            
            # Monitor job progress
            print("   🔄 Monitoring job progress...")
            max_wait = 30  # Maximum wait time in seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status = engine.get_job_status(job_id)
                print(f"   📊 Status: {status['status']} | Progress: {status['progress']*100:.1f}%")
                
                if status['status'] in ['completed', 'failed']:
                    break
                
                time.sleep(1)
            
            # Check final status
            final_status = engine.get_job_status(job_id)
            print(f"   ✅ Final status: {final_status['status']}")
            
            if final_status['status'] == 'completed':
                print(f"   ✅ Duration: {final_status['duration_seconds']:.2f}s")
                print(f"   ✅ Output path: {final_status['output_path']}")
                
                # Verify output file exists
                if final_status['output_path'] and Path(final_status['output_path']).exists():
                    # Read and validate generated data
                    generated_df = pd.read_csv(final_status['output_path'])
                    print(f"   ✅ Generated data shape: {generated_df.shape}")
                    print(f"   ✅ Generated columns: {list(generated_df.columns)}")
                else:
                    print(f"   ❌ Output file not found")
                    return False
            else:
                print(f"   ❌ Job failed: {final_status.get('error_message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"   ❌ Job execution test failed: {e}")
            return False
        
        # Test 6: Multiple Jobs and Resource Management
        print("\n🧪 Test 6: Multiple Jobs and Concurrency")
        try:
            # Create multiple jobs
            jobs = []
            for i in range(3):
                job = engine.create_job(
                    model_name="mock_ctgan",
                    data_path=data_path,
                    n_samples=25,
                    model_config={"epochs": 3},
                    priority=JobPriority.NORMAL
                )
                jobs.append(job)
            
            print(f"   ✅ Created {len(jobs)} jobs")
            
            # Submit jobs
            submitted_ids = []
            for i, job in enumerate(jobs[:2]):  # Submit only 2 (within concurrent limit)
                try:
                    job_id = engine.submit_job(job)
                    submitted_ids.append(job_id)
                    print(f"   ✅ Job {i+1} submitted: {job_id}")
                except Exception as e:
                    print(f"   ⚠️  Job {i+1} submission limited: {e}")
            
            # Try to submit third job (should hit limit)
            try:
                third_job_id = engine.submit_job(jobs[2])
                print(f"   ⚠️  Third job unexpectedly accepted: {third_job_id}")
            except Exception as e:
                print(f"   ✅ Concurrent limit properly enforced: Resource limit reached")
            
            # Wait for jobs to complete
            print("   🔄 Waiting for jobs to complete...")
            completed_count = 0
            max_wait = 45
            start_time = time.time()
            
            while time.time() - start_time < max_wait and completed_count < len(submitted_ids):
                completed_count = 0
                for job_id in submitted_ids:
                    status = engine.get_job_status(job_id)
                    if status['status'] in ['completed', 'failed']:
                        completed_count += 1
                
                if completed_count < len(submitted_ids):
                    time.sleep(2)
            
            print(f"   ✅ {completed_count}/{len(submitted_ids)} jobs completed")
            
        except Exception as e:
            print(f"   ❌ Concurrency test failed: {e}")
            return False
        
        # Test 7: Job Management Operations
        print("\n🧪 Test 7: Job Management Operations")
        try:
            # List all jobs
            all_jobs = engine.list_jobs()
            print(f"   ✅ Total jobs in history: {len(all_jobs)}")
            
            # List jobs by status
            completed_jobs = engine.list_jobs(status=JobStatus.COMPLETED)
            print(f"   ✅ Completed jobs: {len(completed_jobs)}")
            
            # Test job cancellation (create a new job to cancel)
            cancel_job = engine.create_job(
                model_name="mock_ctgan",
                data_path=data_path,
                n_samples=10,
                model_config={"epochs": 1}
            )
            
            # Cancel before submission
            cancelled = engine.cancel_job(cancel_job.job_id)
            print(f"   ✅ Job cancellation: {cancelled}")
            
            # Get engine statistics
            stats = engine.get_engine_stats()
            print(f"   ✅ Engine stats:")
            print(f"       Total jobs: {stats['total_jobs']}")
            print(f"       Success rate: {stats['success_rate']:.1f}%")
            print(f"       Registered models: {stats['registered_models']}")
            
        except Exception as e:
            print(f"   ❌ Job management test failed: {e}")
            return False
        
        # Test 8: Progress Callbacks
        print("\n🧪 Test 8: Progress Callbacks")
        try:
            progress_updates = []
            
            def progress_callback(progress: float, message: str):
                progress_updates.append((progress, message))
                print(f"   📈 Progress: {progress*100:.1f}% - {message}")
            
            # Create job with callback
            callback_job = engine.create_job(
                model_name="mock_ctgan",
                data_path=data_path,
                n_samples=20,
                model_config={"epochs": 2}
            )
            
            # Add callback and submit
            engine.add_progress_callback(callback_job.job_id, progress_callback)
            engine.submit_job(callback_job)
            
            # Wait for completion
            max_wait = 15
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status = engine.get_job_status(callback_job.job_id)
                if status['status'] in ['completed', 'failed']:
                    break
                time.sleep(0.5)
            
            print(f"   ✅ Progress callbacks received: {len(progress_updates)}")
            
        except Exception as e:
            print(f"   ❌ Progress callback test failed: {e}")
            return False
        
        # Test 9: Error Handling
        print("\n🧪 Test 9: Error Handling")
        try:
            # Test invalid model name
            try:
                invalid_job = engine.create_job(
                    model_name="nonexistent_model",
                    data_path=data_path,
                    n_samples=10
                )
                print(f"   ❌ Should have failed with invalid model")
                return False
            except Exception:
                print(f"   ✅ Invalid model properly rejected")
            
            # Test invalid job ID
            try:
                status = engine.get_job_status("invalid_job_id")
                print(f"   ❌ Should have failed with invalid job ID")
                return False
            except Exception:
                print(f"   ✅ Invalid job ID properly rejected")
            
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            return False
        
        # Test 10: Engine Shutdown
        print("\n🧪 Test 10: Engine Shutdown")
        try:
            # Get final stats
            final_stats = engine.get_engine_stats()
            print(f"   ✅ Final engine statistics:")
            print(f"       Total jobs processed: {final_stats['total_jobs']}")
            print(f"       Success rate: {final_stats['success_rate']:.1f}%")
            print(f"       Active jobs: {final_stats['active_jobs']}")
            
            # Shutdown engine
            engine.shutdown()
            print(f"   ✅ SerenityEngine shutdown completed")
            
        except Exception as e:
            print(f"   ❌ Engine shutdown test failed: {e}")
            return False
        
        # Cleanup
        try:
            Path(data_path).unlink()
            print(f"   🧹 Temporary files cleaned up")
        except Exception:
            pass
        
        print(f"\n💙 All SerenityEngine tests passed successfully!")
        print(f"📊 Total test categories: 10")
        print(f"🌟 SerenityEngine is ready to orchestrate ML generation!")
        print(f"💝 Named after Serenity - bringing calm reliability to synthetic data generation")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SerenityEngine: {e}")
        return False
    except Exception as e:
        print(f"❌ SerenityEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_serenity_engine()
    sys.exit(0 if success else 1)