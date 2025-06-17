"""
Core configuration management for Mirage ML Engine.
Optimized for Apple Silicon M4 Pro with MPS acceleration.
"""

import os
import platform
from pathlib import Path
from typing import Optional, Literal

import torch
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class MirageSettings(BaseSettings):
    """Main configuration class for Mirage ML Engine."""
    
    # Environment
    environment: Literal["development", "production", "testing"] = "development"
    debug: bool = Field(default=True, description="Enable debug mode")
    
    # API Configuration
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8001, description="API port")
    workers: int = Field(default=1, description="Number of workers")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    reload: bool = Field(default=True, description="Enable auto-reload")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    
    # Storage Configuration
    storage_path: Path = Field(default=Path("./storage"), description="Base storage path")
    max_upload_size_mb: int = Field(default=100, description="Maximum upload size in MB")
    cleanup_after_hours: int = Field(default=24, description="Clean up temporary files after hours")
    
    # Model Configuration
    default_model: str = Field(default="ctgan", description="Default model to use")
    model_cache_size: int = Field(default=3, description="Number of models to keep in memory")
    max_concurrent_jobs: int = Field(default=5, description="Maximum concurrent generation jobs")
    
    # Hardware Optimization (M4 Pro specific)
    use_mps: bool = Field(default=True, description="Use Apple MPS acceleration")
    mps_fallback: bool = Field(default=True, description="Fallback to CPU if MPS fails")
    torch_threads: int = Field(default=8, description="PyTorch thread count")
    
    # Security
    cors_origins: list[str] = Field(default=["http://localhost:3000"], description="CORS allowed origins")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("storage_path")
    def create_storage_directories(cls, v: Path) -> Path:
        """Ensure storage directories exist."""
        v = v.resolve()
        
        # Create main storage directories
        directories = [
            v / "models",
            v / "datasets", 
            v / "outputs",
            v / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        return v
    
    @validator("torch_threads")
    def validate_torch_threads(cls, v: int) -> int:
        """Validate and set optimal thread count for M4 Pro."""
        # M4 Pro has 12 cores (8 performance + 4 efficiency)
        # For ML workloads, use performance cores only
        max_threads = os.cpu_count() or 8
        optimal_threads = min(v, max_threads, 8)
        return optimal_threads


class SystemInfo:
    """System information and hardware detection."""
    
    def __init__(self):
        self._system = platform.system()
        self._machine = platform.machine()
        self._python_version = platform.python_version()
        
    @property
    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        return self._system == "Darwin" and self._machine == "arm64"
    
    @property
    def is_m4_pro(self) -> bool:
        """Detect if running on M4 Pro (best effort detection)."""
        if not self.is_apple_silicon:
            return False
            
        # Check for M4 Pro indicators
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            cpu_info = result.stdout.strip().lower()
            return "m4" in cpu_info
        except Exception:
            # Fallback: assume modern Apple Silicon
            return True
    
    @property
    def mps_available(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        try:
            return torch.backends.mps.is_available()
        except AttributeError:
            # Older PyTorch versions don't have MPS
            return False
    
    @property
    def mps_built(self) -> bool:
        """Check if PyTorch was built with MPS support."""
        try:
            return torch.backends.mps.is_built()
        except AttributeError:
            return False
    
    def get_optimal_device(self, use_mps: bool = True, mps_fallback: bool = True) -> torch.device:
        """Get the optimal PyTorch device for this system."""
        if use_mps and self.mps_available and self.mps_built:
            try:
                # Test MPS device creation
                device = torch.device("mps")
                # Quick test to ensure MPS works
                torch.tensor([1.0], device=device)
                return device
            except Exception as e:
                if not mps_fallback:
                    raise RuntimeError(f"MPS device failed and fallback disabled: {e}")
                
        # Fallback to CPU
        return torch.device("cpu")
    
    def configure_torch_optimizations(self, settings: MirageSettings) -> None:
        """Configure PyTorch for optimal M4 Pro performance."""
        # Set thread count
        torch.set_num_threads(settings.torch_threads)
        
        # Enable optimizations for Apple Silicon
        if self.is_apple_silicon:
            # Use optimized BLAS operations
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            
            # Memory optimization
            if self.mps_available:
                # MPS-specific optimizations
                torch.mps.empty_cache()  # Clear MPS cache
        
        # General PyTorch optimizations
        torch.set_float32_matmul_precision('medium')  # Balance speed/precision


class MirageConfig:
    """Main configuration manager for Mirage."""
    
    def __init__(self):
        self.settings = MirageSettings()
        self.system = SystemInfo()
        self._device: Optional[torch.device] = None
        
        # Initialize system optimizations
        self._setup_optimizations()
    
    def _setup_optimizations(self) -> None:
        """Set up system-specific optimizations."""
        self.system.configure_torch_optimizations(self.settings)
    
    @property
    def device(self) -> torch.device:
        """Get the optimal PyTorch device (cached)."""
        if self._device is None:
            self._device = self.system.get_optimal_device(
                use_mps=self.settings.use_mps,
                mps_fallback=self.settings.mps_fallback
            )
        return self._device
    
    @property
    def storage_paths(self) -> dict[str, Path]:
        """Get all storage paths."""
        base = self.settings.storage_path
        return {
            "base": base,
            "models": base / "models",
            "datasets": base / "datasets", 
            "outputs": base / "outputs",
            "logs": base / "logs"
        }
    
    def get_model_path(self, model_name: str, version: str = "latest") -> Path:
        """Get path for a specific model version."""
        return self.storage_paths["models"] / model_name / version
    
    def get_system_info(self) -> dict:
        """Get comprehensive system information for debugging."""
        return {
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(), 
                "cpu_count": os.cpu_count(),
                "is_apple_silicon": self.system.is_apple_silicon,
                "is_m4_pro": self.system.is_m4_pro,
            },
            "pytorch": {
                "version": torch.__version__,
                "mps_available": self.system.mps_available,
                "mps_built": self.system.mps_built,
                "device": str(self.device),
                "thread_count": torch.get_num_threads(),
            },
            "storage": {
                "base_path": str(self.settings.storage_path),
                "paths_exist": all(p.exists() for p in self.storage_paths.values()),
            },
            "settings": {
                "environment": self.settings.environment,
                "debug": self.settings.debug,
                "max_concurrent_jobs": self.settings.max_concurrent_jobs,
            }
        }


# Global configuration instance
config = MirageConfig()

# Convenience exports
settings = config.settings
device = config.device
storage_paths = config.storage_paths


def get_system_info() -> dict:
    """Quick access to system info."""
    return config.get_system_info()


# Development helper
if __name__ == "__main__":
    import json
    print("🔥 Mirage Configuration Test")
    print("=" * 50)
    print(json.dumps(get_system_info(), indent=2, default=str))