"""
Mirage Models Package
Production-grade synthetic data models for all data types.
"""

# Import all tabular models
from .tabular.ctgan import CTGANGenerator, CTGANConfig
from .tabular.tvae import TVAEGenerator, TVAEConfig
from .tabular.copulagan import CopulaGANGenerator, CopulaGANConfig

# Import vision models (when implemented)
# from .vision.diffusion import DiffusionGenerator, DiffusionConfig

# Export all models
__all__ = [
    # Tabular models
    "CTGANGenerator",
    "CTGANConfig",
    "TVAEGenerator", 
    "TVAEConfig",
    "CopulaGANGenerator",
    "CopulaGANConfig",
    
    # Vision models (future)
    # "DiffusionGenerator",
    # "DiffusionConfig",
]