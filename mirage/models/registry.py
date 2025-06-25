"""
Model Registry for Mirage ML Engine
Centralized registry for all available synthetic data models.
"""

from typing import Dict, Type, Any, List, Optional
from enum import Enum
from loguru import logger

from ..core.base import BaseGenerator, BaseGeneratorConfig, DataType
from ..core.exceptions import ModelError

# Import all available models
from .tabular.ctgan import CTGANGenerator, CTGANConfig
from .tabular.tvae import TVAEGenerator, TVAEConfig
from .tabular.copulagan import CopulaGANGenerator, CopulaGANConfig


class ModelType(str, Enum):
    """Available model types."""
    CTGAN = "ctgan"
    TVAE = "tvae"
    COPULAGAN = "copulagan"
    DIFFUSION = "diffusion"  # Future implementation


class ModelRegistry:
    """
    Registry for managing all available synthetic data models.
    
    Provides a centralized way to:
    - Register new models
    - Create model instances
    - List available models
    - Get model configurations
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._generators: Dict[str, Type[BaseGenerator]] = {}
        self._configs: Dict[str, Type[BaseGeneratorConfig]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Register all available models
        self._register_built_in_models()
        
        logger.info(f"=' Model registry initialized with {len(self._generators)} models")
    
    def _register_built_in_models(self) -> None:
        """Register all built-in models."""
        
        # Register CTGAN
        self.register_model(
            model_type=ModelType.CTGAN,
            generator_class=CTGANGenerator,
            config_class=CTGANConfig,
            metadata={
                "name": "CTGAN",
                "description": "Conditional Tabular GAN for synthetic tabular data",
                "data_types": [DataType.TABULAR],
                "strengths": [
                    "Handles mixed data types well",
                    "Good for categorical data", 
                    "Stable training",
                    "Industry standard"
                ],
                "use_cases": [
                    "Healthcare data",
                    "Financial records", 
                    "Customer data",
                    "General tabular data"
                ],
                "healthcare_optimized": True,
                "apple_silicon_optimized": True,
                "privacy_levels": ["standard", "high", "maximum"]
            }
        )
        
        # Register TVAE
        self.register_model(
            model_type=ModelType.TVAE,
            generator_class=TVAEGenerator,
            config_class=TVAEConfig,
            metadata={
                "name": "TVAE", 
                "description": "Tabular Variational Autoencoder for synthetic data",
                "data_types": [DataType.TABULAR],
                "strengths": [
                    "Better for continuous variables",
                    "Smoother latent space",
                    "Good reconstruction quality",
                    "Faster training than GANs"
                ],
                "use_cases": [
                    "Continuous healthcare metrics",
                    "Financial time series",
                    "Scientific measurements", 
                    "Sensor data"
                ],
                "healthcare_optimized": True,
                "apple_silicon_optimized": True,
                "privacy_levels": ["standard", "high", "maximum"]
            }
        )
        
        # Register CopulaGAN
        self.register_model(
            model_type=ModelType.COPULAGAN,
            generator_class=CopulaGANGenerator,
            config_class=CopulaGANConfig,
            metadata={
                "name": "CopulaGAN",
                "description": "Copula-based GAN preserving marginal distributions",
                "data_types": [DataType.TABULAR],
                "strengths": [
                    "Preserves marginal distributions",
                    "Models complex dependencies", 
                    "Handles mixed data types",
                    "Better statistical properties"
                ],
                "use_cases": [
                    "Financial risk modeling",
                    "Insurance data",
                    "Complex healthcare data",
                    "Multi-variate analysis"
                ],
                "healthcare_optimized": True,
                "apple_silicon_optimized": True,
                "privacy_levels": ["standard", "high", "maximum"]
            }
        )
        
        logger.info(" Built-in models registered: CTGAN, TVAE, CopulaGAN")
    
    def register_model(
        self,
        model_type: ModelType,
        generator_class: Type[BaseGenerator],
        config_class: Type[BaseGeneratorConfig],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Unique model type identifier
            generator_class: Generator class
            config_class: Configuration class
            metadata: Optional model metadata
        """
        model_key = model_type.value
        
        # Validate classes
        if not issubclass(generator_class, BaseGenerator):
            raise ModelError(f"Generator class must inherit from BaseGenerator: {generator_class}")
        
        if not issubclass(config_class, BaseGeneratorConfig):
            raise ModelError(f"Config class must inherit from BaseGeneratorConfig: {config_class}")
        
        # Register the model
        self._generators[model_key] = generator_class
        self._configs[model_key] = config_class
        self._metadata[model_key] = metadata or {}
        
        logger.info(f"=Ý Registered model: {model_key}")
    
    def create_generator(
        self,
        model_type: ModelType,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseGenerator:
        """
        Create a generator instance.
        
        Args:
            model_type: Type of model to create
            config: Optional configuration parameters
            
        Returns:
            Configured generator instance
            
        Raises:
            ModelError: If model type is not registered
        """
        model_key = model_type.value
        
        if model_key not in self._generators:
            available = list(self._generators.keys())
            raise ModelError(
                f"Model type '{model_key}' not registered. Available: {available}",
                model_name=model_key
            )
        
        generator_class = self._generators[model_key]
        config_class = self._configs[model_key]
        
        # Create configuration instance
        if config:
            model_config = config_class(**config)
        else:
            model_config = config_class(
                name=model_key,
                data_type=DataType.TABULAR  # Default for all current models
            )
        
        # Create and return generator
        generator = generator_class(model_config)
        
        logger.info(f"<×  Created {model_key} generator: {generator.config.name}")
        return generator
    
    def get_available_models(self, data_type: Optional[DataType] = None) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Args:
            data_type: Optional filter by data type
            
        Returns:
            List of model information
        """
        models = []
        
        for model_key, metadata in self._metadata.items():
            # Filter by data type if specified
            if data_type:
                model_data_types = metadata.get("data_types", [])
                if data_type not in model_data_types:
                    continue
            
            model_info = {
                "type": model_key,
                "name": metadata.get("name", model_key.upper()),
                "description": metadata.get("description", ""),
                "data_types": metadata.get("data_types", []),
                "strengths": metadata.get("strengths", []),
                "use_cases": metadata.get("use_cases", []),
                "healthcare_optimized": metadata.get("healthcare_optimized", False),
                "apple_silicon_optimized": metadata.get("apple_silicon_optimized", False),
                "privacy_levels": metadata.get("privacy_levels", [])
            }
            models.append(model_info)
        
        return models
    
    def get_model_config_schema(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get configuration schema for a model type.
        
        Args:
            model_type: Model type
            
        Returns:
            JSONSchema for model configuration
            
        Raises:
            ModelError: If model type is not registered
        """
        model_key = model_type.value
        
        if model_key not in self._generators:
            available = list(self._generators.keys())
            raise ModelError(
                f"Model type '{model_key}' not registered. Available: {available}",
                model_name=model_key
            )
        
        # Create temporary instance to get schema
        generator_class = self._generators[model_key]
        config_class = self._configs[model_key]
        
        # Create minimal config for schema extraction
        temp_config = config_class(
            name=f"temp_{model_key}",
            data_type=DataType.TABULAR
        )
        temp_generator = generator_class(temp_config)
        
        return temp_generator.get_config_schema()
    
    def get_model_metadata(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get metadata for a model type.
        
        Args:
            model_type: Model type
            
        Returns:
            Model metadata
        """
        model_key = model_type.value
        return self._metadata.get(model_key, {})
    
    def is_registered(self, model_type: ModelType) -> bool:
        """Check if a model type is registered."""
        return model_type.value in self._generators
    
    def get_healthcare_models(self) -> List[Dict[str, Any]]:
        """Get all healthcare-optimized models."""
        return [
            model for model in self.get_available_models()
            if model.get("healthcare_optimized", False)
        ]
    
    def get_apple_silicon_models(self) -> List[Dict[str, Any]]:
        """Get all Apple Silicon optimized models.""" 
        return [
            model for model in self.get_available_models()
            if model.get("apple_silicon_optimized", False)
        ]
    
    def recommend_model(
        self,
        data_type: DataType,
        use_case: Optional[str] = None,
        data_characteristics: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Recommend models based on requirements.
        
        Args:
            data_type: Type of data to synthesize
            use_case: Specific use case (e.g., "healthcare", "financial")
            data_characteristics: Data characteristics (e.g., {"categorical_heavy": True})
            
        Returns:
            List of recommended model types (ordered by preference)
        """
        recommendations = []
        available_models = self.get_available_models(data_type)
        
        for model in available_models:
            score = 0
            
            # Base score for supporting the data type
            if data_type in model["data_types"]:
                score += 1
            
            # Use case matching
            if use_case:
                use_case_lower = use_case.lower()
                model_use_cases = [uc.lower() for uc in model["use_cases"]]
                if any(use_case_lower in uc for uc in model_use_cases):
                    score += 2
            
            # Healthcare optimization bonus
            if use_case and "health" in use_case.lower() and model["healthcare_optimized"]:
                score += 1
            
            # Apple Silicon optimization bonus (always beneficial)
            if model["apple_silicon_optimized"]:
                score += 1
            
            recommendations.append((score, model["type"]))
        
        # Sort by score (descending) and return model types
        recommendations.sort(key=lambda x: x[0], reverse=True)
        return [model_type for score, model_type in recommendations]


# Global registry instance
model_registry = ModelRegistry()

# Convenience functions
def create_generator(model_type: ModelType, config: Optional[Dict[str, Any]] = None) -> BaseGenerator:
    """Create a generator using the global registry."""
    return model_registry.create_generator(model_type, config)

def get_available_models(data_type: Optional[DataType] = None) -> List[Dict[str, Any]]:
    """Get available models using the global registry.""" 
    return model_registry.get_available_models(data_type)

def recommend_model(
    data_type: DataType,
    use_case: Optional[str] = None, 
    data_characteristics: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Get model recommendations using the global registry."""
    return model_registry.recommend_model(data_type, use_case, data_characteristics)


# Export all public components
__all__ = [
    "ModelType",
    "ModelRegistry", 
    "model_registry",
    "create_generator",
    "get_available_models",
    "recommend_model"
]