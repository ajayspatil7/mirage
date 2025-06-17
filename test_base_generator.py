#!/usr/bin/env python3
"""
Test script for Mirage BaseGenerator interface.
Run this to verify the base generator system works correctly.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Union, Optional

import torch
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class MockGenerator:
    """Mock implementation of BaseGenerator for testing."""
    
    def __init__(self):
        # Import here to avoid circular imports during testing
        from mirage.core.base import BaseGenerator, BaseGeneratorConfig, DataType
        
        # Create a test configuration
        config = BaseGeneratorConfig(
            name="test_generator",
            version="1.0.0",
            description="Test generator for unit testing",
            data_type=DataType.TABULAR,
            batch_size=32,
            random_seed=42
        )
        
        # Store config and create proper inheritance
        self.config = config
        self._setup_base_generator()
        
        # Test data
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randint(0, 5, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    def _setup_base_generator(self):
        """Setup base generator functionality."""
        from mirage.core.base import BaseGenerator, ModelStatus
        
        # Create a concrete implementation
        class ConcreteGenerator(BaseGenerator):
            def get_config_schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "epochs": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "learning_rate": {"type": "number", "minimum": 0.0001, "maximum": 1.0}
                    },
                    "required": ["epochs"]
                }
            
            def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
                validated = config.copy()
                validated.setdefault("epochs", 100)
                validated.setdefault("learning_rate", 0.001)
                
                if not 1 <= validated["epochs"] <= 1000:
                    raise ValueError("epochs must be between 1 and 1000")
                if not 0.0001 <= validated["learning_rate"] <= 1.0:
                    raise ValueError("learning_rate must be between 0.0001 and 1.0")
                    
                return validated
            
            def fit(self, data, validation_data=None, **kwargs):
                from mirage.core.base import ModelStatus, TrainingProgress
                
                self._set_status(ModelStatus.TRAINING)
                self.validate_data(data)
                
                if isinstance(data, pd.DataFrame):
                    self._training_data_shape = data.shape
                    self._column_names = list(data.columns)
                elif isinstance(data, np.ndarray):
                    self._training_data_shape = data.shape
                
                epochs = kwargs.get('epochs', 5)
                for epoch in range(1, epochs + 1):
                    progress = TrainingProgress(
                        epoch=epoch,
                        total_epochs=epochs,
                        loss=1.0 / epoch,
                        elapsed_time=epoch * 0.1
                    )
                    self._update_training_progress(progress)
                
                self.metadata.training_epochs = epochs
                self.metadata.training_samples = len(data)
                self.metadata.training_duration_seconds = epochs * 0.1
                
                self._set_status(ModelStatus.TRAINED)
            
            def generate(self, n_samples: int, **kwargs):
                from mirage.core.base import ModelStatus
                
                if not self.is_trained:
                    raise RuntimeError("Model must be trained before generating")
                
                self._set_status(ModelStatus.GENERATING)
                
                if self._column_names:
                    generated_data = pd.DataFrame({
                        col: np.random.randn(n_samples) for col in self._column_names
                    })
                else:
                    shape = (n_samples,) + self._training_data_shape[1:]
                    generated_data = np.random.randn(*shape)
                
                self._set_status(ModelStatus.TRAINED)
                return generated_data
            
            def _save_model_data(self, path: Path) -> None:
                model_file = path / "model.pt"
                torch.save({"dummy": "model"}, model_file)
            
            def _load_model_data(self, path: Path) -> None:
                model_file = path / "model.pt"
                if model_file.exists():
                    torch.load(model_file)
        
        # Create the concrete generator instance
        self.generator = ConcreteGenerator(self.config)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the generator."""
        return getattr(self.generator, name)


def test_base_generator():
    """Test the BaseGenerator interface."""
    print("🔥 Testing Mirage BaseGenerator Interface")
    print("=" * 50)
    
    try:
        # Import all required classes
        from mirage.core.base import (
            BaseGenerator,
            BaseGeneratorConfig,
            ModelMetadata,
            TrainingProgress,
            ModelStatus,
            DataType,
            convert_to_tensor,
            convert_from_tensor
        )
        
        print("✅ All base generator classes imported successfully")
        
        # Test 1: Configuration and Metadata
        print("\n🧪 Test 1: Configuration and Metadata")
        try:
            config = BaseGeneratorConfig(
                name="test_model",
                version="1.0.0",
                data_type=DataType.TABULAR,
                batch_size=64,
                random_seed=123
            )
            print(f"   ✅ Configuration created: {config.name}")
            print(f"   ✅ Data type: {config.data_type}")
            print(f"   ✅ Device: {config.device}")
            
            metadata = ModelMetadata(
                name="test_model",
                version="1.0.0",
                data_type=DataType.TABULAR,
                device_used="mps",
                python_version="3.11.9"
            )
            print(f"   ✅ Metadata created: {metadata.name}")
            
        except Exception as e:
            print(f"   ❌ Configuration test failed: {e}")
            return False
        
        # Test 2: Mock Generator Creation
        print("\n🧪 Test 2: Mock Generator Creation")
        try:
            mock_gen = MockGenerator()
            
            print(f"   ✅ Generator created: {mock_gen}")
            print(f"   ✅ Status: {mock_gen.status}")
            print(f"   ✅ Device: {mock_gen.device}")
            print(f"   ✅ Is trained: {mock_gen.is_trained}")
            
        except Exception as e:
            print(f"   ❌ Generator creation failed: {e}")
            return False
        
        # Test 3: Data Validation
        print("\n🧪 Test 3: Data Validation")
        try:
            mock_gen = MockGenerator()
            
            # Test valid data
            valid_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            valid_array = np.array([[1, 2], [3, 4], [5, 6]])
            valid_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            
            mock_gen.validate_data(valid_df)
            mock_gen.validate_data(valid_array)
            mock_gen.validate_data(valid_tensor)
            print("   ✅ Valid data validation passed")
            
            # Test invalid data
            try:
                mock_gen.validate_data(None)
                print("   ❌ Should have failed on None data")
                return False
            except Exception:
                print("   ✅ Correctly rejected None data")
            
            try:
                mock_gen.validate_data(pd.DataFrame())
                print("   ❌ Should have failed on empty DataFrame")
                return False
            except Exception:
                print("   ✅ Correctly rejected empty DataFrame")
                
        except Exception as e:
            print(f"   ❌ Data validation test failed: {e}")
            return False
        
        # Test 4: Training
        print("\n🧪 Test 4: Training Process")
        try:
            # Train the mock generator
            mock_gen = MockGenerator()
            training_data = mock_gen.test_data
            mock_gen.fit(training_data, epochs=3)
            
            print(f"   ✅ Training completed")
            print(f"   ✅ Status after training: {mock_gen.status}")
            print(f"   ✅ Is trained: {mock_gen.is_trained}")
            print(f"   ✅ Training data shape: {mock_gen.training_data_shape}")
            print(f"   ✅ Column names: {mock_gen.column_names}")
            
        except Exception as e:
            print(f"   ❌ Training test failed: {e}")
            return False
        
        # Test 5: Generation
        print("\n🧪 Test 5: Data Generation")
        try:
            # Generate synthetic data
            n_samples = 10
            generated = mock_gen.generate(n_samples)
            
            print(f"   ✅ Generated {len(generated)} samples")
            print(f"   ✅ Generated data type: {type(generated)}")
            if hasattr(generated, 'shape'):
                print(f"   ✅ Generated data shape: {generated.shape}")
            if hasattr(generated, 'columns'):
                print(f"   ✅ Generated columns: {list(generated.columns)}")
                
        except Exception as e:
            print(f"   ❌ Generation test failed: {e}")
            return False
        
        # Test 6: Model Save/Load
        print("\n🧪 Test 6: Model Save/Load")
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / "test_model"
                
                # Save model
                mock_gen.save_model(model_path)
                print(f"   ✅ Model saved to: {model_path}")
                
                # Check files exist
                assert (model_path / "metadata.json").exists()
                assert (model_path / "config.json").exists()
                assert (model_path / "model.pt").exists()
                print("   ✅ All model files created")
                
                # Create new generator and load
                new_mock_gen = MockGenerator()
                
                new_mock_gen.load_model(model_path)
                print("   ✅ Model loaded successfully")
                print(f"   ✅ Loaded status: {new_mock_gen.status}")
                print(f"   ✅ Loaded is_trained: {new_mock_gen.is_trained}")
                
        except Exception as e:
            print(f"   ❌ Save/Load test failed: {e}")
            return False
        
        # Test 7: Utility Functions
        print("\n🧪 Test 7: Utility Functions")
        try:
            mock_gen = MockGenerator()
            
            # Test tensor conversion
            test_df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
            test_array = np.array([[1.0, 2.0], [3.0, 4.0]])
            
            # Convert to tensor
            tensor_from_df = convert_to_tensor(test_df, mock_gen.device)
            tensor_from_array = convert_to_tensor(test_array, mock_gen.device)
            
            print(f"   ✅ DataFrame to tensor: {tensor_from_df.shape}")
            print(f"   ✅ Array to tensor: {tensor_from_array.shape}")
            print(f"   ✅ Tensor device: {tensor_from_df.device}")
            
            # Convert from tensor
            array_from_tensor = convert_from_tensor(tensor_from_df, "numpy")
            df_from_tensor = convert_from_tensor(tensor_from_df, "pandas", ["col1", "col2"])
            
            print(f"   ✅ Tensor to array: {array_from_tensor.shape}")
            print(f"   ✅ Tensor to DataFrame: {df_from_tensor.shape}")
            print(f"   ✅ DataFrame columns: {list(df_from_tensor.columns)}")
            
        except Exception as e:
            print(f"   ❌ Utility functions test failed: {e}")
            return False
        
        # Test 8: Model Info and Memory Estimation
        print("\n🧪 Test 8: Model Info and Memory Estimation")
        try:
            # Get model info
            model_info = mock_gen.get_model_info()
            
            print(f"   ✅ Model info retrieved")
            print(f"   ✅ Name: {model_info['name']}")
            print(f"   ✅ Status: {model_info['status']}")
            print(f"   ✅ Device: {model_info['device']}")
            print(f"   ✅ Is trained: {model_info['is_trained']}")
            
            # Test memory estimation
            memory_estimate = mock_gen.estimate_memory_usage(1000)
            print(f"   ✅ Memory estimate for 1000 samples: {memory_estimate:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ Model info test failed: {e}")
            return False
        
        # Test 9: Training Progress
        print("\n🧪 Test 9: Training Progress")
        try:
            progress = TrainingProgress(
                epoch=5,
                total_epochs=10,
                loss=0.123,
                elapsed_time=30.5
            )
            
            print(f"   ✅ Progress created: {progress.progress_percent:.1f}%")
            print(f"   ✅ Epoch: {progress.epoch}/{progress.total_epochs}")
            print(f"   ✅ Loss: {progress.loss}")
            
        except Exception as e:
            print(f"   ❌ Training progress test failed: {e}")
            return False
        
        # Test 10: Error Handling
        print("\n🧪 Test 10: Error Handling")
        try:
            # Test generation without training
            untrained_mock = MockGenerator()
            
            try:
                untrained_mock.generate(10)
                print("   ❌ Should have failed on untrained model")
                return False
            except Exception:
                print("   ✅ Correctly rejected generation on untrained model")
            
            # Test save without training
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    untrained_mock.save_model(Path(temp_dir) / "model")
                print("   ❌ Should have failed saving untrained model")
                return False
            except Exception:
                print("   ✅ Correctly rejected saving untrained model")
            
            # Test load from non-existent path
            try:
                untrained_mock.load_model(Path("/nonexistent/path"))
                print("   ❌ Should have failed loading from non-existent path")
                return False
            except Exception:
                print("   ✅ Correctly rejected loading from non-existent path")
                
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            return False
        
        print(f"\n🚀 All BaseGenerator tests passed successfully!")
        print(f"📊 Total test categories: 10")
        print(f"🏗️  BaseGenerator interface ready for implementation!")
        print(f"🎯 Ready for CTGAN, TVAE, and other model implementations!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import base generator: {e}")
        return False
    except Exception as e:
        print(f"❌ BaseGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_base_generator()
    sys.exit(0 if success else 1)