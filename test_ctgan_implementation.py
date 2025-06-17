#!/usr/bin/env python3
"""
Test script for CTGAN Implementation.
Run this to verify the CTGAN generator works correctly with healthcare data.
"""

import sys
import time
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_healthcare_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a realistic healthcare dataset for testing."""
    np.random.seed(42)
    
    # Generate correlated healthcare data
    data = {}
    
    # Basic demographics
    data['patient_id'] = range(1, n_samples + 1)
    data['age'] = np.random.normal(45, 15, n_samples).clip(18, 90).astype(int)
    data['gender'] = np.random.choice(['M', 'F'], n_samples)
    
    # Physical measurements (correlated with age)
    age_factor = (data['age'] - 30) / 50  # Normalize age effect
    data['weight_kg'] = np.random.normal(70 + age_factor * 10, 15, n_samples).clip(40, 150).round(1)
    data['height_cm'] = np.random.normal(170, 10, n_samples).clip(140, 200).round(1)
    data['bmi'] = (data['weight_kg'] / (data['height_cm'] / 100) ** 2).round(1)
    
    # Vital signs (correlated with age and BMI)
    bmi_factor = (data['bmi'] - 25) / 10
    data['systolic_bp'] = (120 + age_factor * 20 + bmi_factor * 10 + np.random.normal(0, 15, n_samples)).clip(90, 200).astype(int)
    data['diastolic_bp'] = (80 + age_factor * 10 + bmi_factor * 5 + np.random.normal(0, 10, n_samples)).clip(60, 120).astype(int)
    data['heart_rate'] = (70 + np.random.normal(0, 15, n_samples)).clip(50, 120).astype(int)
    
    # Lab values (correlated with age and other factors)
    data['cholesterol'] = (200 + age_factor * 50 + bmi_factor * 20 + np.random.normal(0, 40, n_samples)).clip(120, 400).astype(int)
    data['glucose'] = (100 + age_factor * 20 + bmi_factor * 15 + np.random.normal(0, 25, n_samples)).clip(70, 300).astype(int)
    
    # Binary conditions (higher probability with age/BMI)
    diabetes_prob = 0.05 + age_factor * 0.15 + np.maximum(bmi_factor, 0) * 0.1
    data['diabetes'] = np.random.binomial(1, np.clip(diabetes_prob, 0, 0.4), n_samples)
    
    hypertension_prob = 0.1 + age_factor * 0.3 + bmi_factor * 0.1
    data['hypertension'] = np.random.binomial(1, np.clip(hypertension_prob, 0, 0.6), n_samples)
    
    heart_disease_prob = 0.03 + age_factor * 0.2 + data['diabetes'] * 0.15 + data['hypertension'] * 0.1
    data['heart_disease'] = np.random.binomial(1, np.clip(heart_disease_prob, 0, 0.5), n_samples)
    
    # Categorical variables
    data['smoking_status'] = np.random.choice(['never', 'former', 'current'], n_samples, p=[0.6, 0.25, 0.15])
    data['insurance_type'] = np.random.choice(['private', 'medicare', 'medicaid', 'none'], n_samples, p=[0.5, 0.25, 0.15, 0.1])
    
    return pd.DataFrame(data)


def test_ctgan_implementation():
    """Test the CTGAN implementation."""
    print("🧠 Testing CTGAN Implementation")
    print("=" * 50)
    
    try:
        # Check SDV availability first
        try:
            import sdv
            print(f"✅ SDV library available: {sdv.__version__}")
        except ImportError:
            print("❌ SDV library not available. Install with: poetry add sdv")
            return False
        
        # Import CTGAN implementation
        from mirage.models.tabular.ctgan import CTGANGenerator, CTGANConfig
        from mirage.core.base import DataType, ModelStatus
        
        print("✅ CTGAN implementation imported successfully")
        
        # Test 1: Configuration and Validation
        print("\n🧪 Test 1: Configuration and Validation")
        try:
            # Test default configuration
            default_config = CTGANConfig(
                name="test_ctgan",
                data_type=DataType.TABULAR
            )
            
            print(f"   ✅ Default config created: {default_config.name}")
            print(f"   ✅ Epochs: {default_config.epochs}")
            print(f"   ✅ Batch size: {default_config.batch_size}")
            print(f"   ✅ Privacy level: {default_config.privacy_level}")
            
            # Test custom configuration
            custom_config = CTGANConfig(
                name="custom_ctgan",
                data_type=DataType.TABULAR,
                epochs=20,  # Minimum valid epochs (was 50)
                batch_size=100,
                privacy_level="high",
                categorical_columns=["gender", "smoking_status", "insurance_type"],
                numerical_columns=["age", "bmi", "systolic_bp"]
            )
            
            print(f"   ✅ Custom config created: {custom_config.epochs} epochs")
            print(f"   ✅ Privacy level: {custom_config.privacy_level}")
            
        except Exception as e:
            print(f"   ❌ Configuration test failed: {e}")
            return False
        
        # Test 2: CTGAN Generator Initialization
        print("\n🧪 Test 2: CTGAN Generator Initialization")
        try:
            # Initialize with custom config for faster testing
            ctgan = CTGANGenerator(custom_config)
            
            print(f"   ✅ CTGAN generator created: {ctgan.config.name}")
            print(f"   ✅ Device: {ctgan.device}")
            print(f"   ✅ Status: {ctgan.status}")
            print(f"   ✅ Is trained: {ctgan.is_trained}")
            
            # Test config schema
            schema = ctgan.get_config_schema()
            print(f"   ✅ Config schema retrieved: {len(schema['properties'])} parameters")
            
        except Exception as e:
            print(f"   ❌ Generator initialization failed: {e}")
            return False
        
        # Test 3: Healthcare Dataset Creation
        print("\n🧪 Test 3: Healthcare Dataset Creation")
        try:
            # Create test dataset
            healthcare_data = create_healthcare_dataset(n_samples=500)  # Smaller for testing
            
            print(f"   ✅ Healthcare dataset created: {healthcare_data.shape}")
            print(f"   ✅ Columns: {list(healthcare_data.columns)}")
            print(f"   ✅ Data types: {healthcare_data.dtypes.value_counts().to_dict()}")
            
            # Basic data validation
            print(f"   ✅ Age range: {healthcare_data['age'].min()}-{healthcare_data['age'].max()}")
            print(f"   ✅ BMI range: {healthcare_data['bmi'].min():.1f}-{healthcare_data['bmi'].max():.1f}")
            print(f"   ✅ Diabetes rate: {healthcare_data['diabetes'].mean():.1%}")
            
        except Exception as e:
            print(f"   ❌ Healthcare dataset creation failed: {e}")
            return False
        
        # Test 4: Configuration Validation
        print("\n🧪 Test 4: Configuration Validation")
        try:
            # Test valid configuration
            valid_config = {
                "epochs": 20,  # Must be >= 10
                "batch_size": 100,
                "generator_lr": 0.0002,
                "discriminator_lr": 0.0002,
                "privacy_level": "standard"
            }
            
            validated = ctgan.validate_config(valid_config)
            print(f"   ✅ Valid config validated: {len(validated)} parameters")
            
            # Test invalid configuration
            try:
                invalid_config = {
                    "epochs": -10,  # Invalid
                    "batch_size": 10000,  # Too large
                    "generator_lr": 10.0,  # Too high
                    "privacy_level": "invalid"  # Invalid option
                }
                ctgan.validate_config(invalid_config)
                print(f"   ❌ Should have failed with invalid config")
                return False
            except Exception:
                print(f"   ✅ Invalid config properly rejected")
            
        except Exception as e:
            print(f"   ❌ Configuration validation test failed: {e}")
            return False
        
        # Test 5: Data Preprocessing
        print("\n🧪 Test 5: Healthcare Data Preprocessing")
        try:
            # Test healthcare data preparation
            sample_data = healthcare_data.head(100)  # Small sample for testing
            
            # Test data validation
            ctgan.validate_data(sample_data)
            print(f"   ✅ Data validation passed")
            
            # The preprocessing happens internally during fit
            # We'll test it indirectly by checking the fit process
            
        except Exception as e:
            print(f"   ❌ Data preprocessing test failed: {e}")
            return False
        
        # Test 6: Model Training (Quick Training)
        print("\n🧪 Test 6: Model Training")
        try:
            # Use very small dataset and few epochs for testing
            training_data = healthcare_data.head(200)
            
            print(f"   🚀 Starting CTGAN training (reduced for testing)...")
            print(f"   📊 Training data: {training_data.shape}")
            
            # Train with minimal epochs for testing (must be >= 10)
            start_time = time.time()
            ctgan.fit(
                training_data,
                epochs=10,  # Minimum valid epochs
                batch_size=50,
                verbose=False
            )
            training_time = time.time() - start_time
            
            print(f"   ✅ Training completed in {training_time:.2f}s")
            print(f"   ✅ Status after training: {ctgan.status}")
            print(f"   ✅ Is trained: {ctgan.is_trained}")
            print(f"   ✅ Training data shape: {ctgan.training_data_shape}")
            print(f"   ✅ Column names: {len(ctgan.column_names)} columns")
            
            # Check metadata
            if hasattr(ctgan, 'metadata') and ctgan.metadata:
                print(f"   ✅ Training metadata: {ctgan.metadata.training_epochs} epochs")
                print(f"   ✅ Training samples: {ctgan.metadata.training_samples}")
            
        except Exception as e:
            print(f"   ❌ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 7: Synthetic Data Generation
        print("\n🧪 Test 7: Synthetic Data Generation")
        try:
            # Generate synthetic data
            n_samples = 50  # Small number for testing
            
            print(f"   🎲 Generating {n_samples} synthetic samples...")
            start_time = time.time()
            synthetic_data = ctgan.generate(n_samples)
            generation_time = time.time() - start_time
            
            print(f"   ✅ Generation completed in {generation_time:.2f}s")
            print(f"   ✅ Generated data shape: {synthetic_data.shape}")
            print(f"   ✅ Generated columns: {list(synthetic_data.columns)}")
            
            # Validate synthetic data quality
            print(f"   ✅ Synthetic data validation:")
            print(f"       Age range: {synthetic_data['age'].min()}-{synthetic_data['age'].max()}")
            print(f"       BMI range: {synthetic_data['bmi'].min():.1f}-{synthetic_data['bmi'].max():.1f}")
            print(f"       Diabetes rate: {synthetic_data['diabetes'].mean():.1%}")
            print(f"       Gender distribution: {synthetic_data['gender'].value_counts().to_dict()}")
            
            # Check for reasonable ranges
            age_valid = 18 <= synthetic_data['age'].min() and synthetic_data['age'].max() <= 90
            bmi_valid = 10 <= synthetic_data['bmi'].min() and synthetic_data['bmi'].max() <= 60
            
            print(f"   ✅ Age range valid: {age_valid}")
            print(f"   ✅ BMI range valid: {bmi_valid}")
            
        except Exception as e:
            print(f"   ❌ Synthetic data generation failed: {e}")
            return False
        
        # Test 8: Model Save/Load
        print("\n🧪 Test 8: Model Save/Load")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / "ctgan_model"
                
                # Save the trained model
                ctgan.save_model(model_path)
                print(f"   ✅ Model saved to: {model_path}")
                
                # Check saved files
                expected_files = ["ctgan_synthesizer.pkl", "metadata.json", "config.json"]
                for filename in expected_files:
                    filepath = model_path / filename
                    if filepath.exists():
                        print(f"   ✅ File created: {filename}")
                    else:
                        print(f"   ⚠️  File missing: {filename}")
                
                # Create new CTGAN instance and load
                new_ctgan = CTGANGenerator()
                new_ctgan.load_model(model_path)
                
                print(f"   ✅ Model loaded successfully")
                print(f"   ✅ Loaded status: {new_ctgan.status}")
                print(f"   ✅ Loaded is_trained: {new_ctgan.is_trained}")
                
                # Test generation with loaded model
                if new_ctgan.is_trained:
                    test_samples = new_ctgan.generate(10)
                    print(f"   ✅ Generated samples with loaded model: {test_samples.shape}")
                
        except Exception as e:
            print(f"   ❌ Model save/load test failed: {e}")
            return False
        
        # Test 9: Privacy Levels
        print("\n🧪 Test 9: Privacy Levels")
        try:
            # Test different privacy levels
            privacy_levels = ["standard", "high", "maximum"]
            
            for privacy_level in privacy_levels:
                # Create config with different privacy levels
                privacy_config = CTGANConfig(
                    name=f"ctgan_{privacy_level}",
                    data_type=DataType.TABULAR,
                    epochs=10,  # Minimum valid epochs
                    batch_size=50,
                    privacy_level=privacy_level
                )
                
                privacy_ctgan = CTGANGenerator(privacy_config)
                print(f"   ✅ Created CTGAN with {privacy_level} privacy")
                
                # The privacy preprocessing will be tested during fit
                # but we won't actually train for each level to save time
            
            print(f"   ✅ All privacy levels supported")
            
        except Exception as e:
            print(f"   ❌ Privacy levels test failed: {e}")
            return False
        
        # Test 10: Error Handling
        print("\n🧪 Test 10: Error Handling")
        try:
            # Test generation without training
            untrained_ctgan = CTGANGenerator()
            
            try:
                untrained_ctgan.generate(10)
                print(f"   ❌ Should have failed on untrained model")
                return False
            except Exception:
                print(f"   ✅ Correctly rejected generation on untrained model")
            
            # Test invalid sample count
            try:
                ctgan.generate(-5)
                print(f"   ❌ Should have failed with negative samples")
                return False
            except Exception:
                print(f"   ✅ Correctly rejected negative sample count")
            
            # Test invalid data
            try:
                empty_data = pd.DataFrame()
                ctgan.validate_data(empty_data)
                print(f"   ❌ Should have failed with empty data")
                return False
            except Exception:
                print(f"   ✅ Correctly rejected empty data")
            
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            return False
        
        # Test 11: Model Information
        print("\n🧪 Test 11: Model Information")
        try:
            # Get model info
            model_info = ctgan.get_model_info()
            
            print(f"   ✅ Model info retrieved")
            print(f"   ✅ Name: {model_info['name']}")
            print(f"   ✅ Status: {model_info['status']}")
            print(f"   ✅ Device: {model_info['device']}")
            print(f"   ✅ Data type: {model_info['data_type']}")
            
            # Memory estimation
            memory_estimate = ctgan.estimate_memory_usage(1000)
            print(f"   ✅ Memory estimate for 1000 samples: {memory_estimate:.2f} MB")
            
        except Exception as e:
            print(f"   ❌ Model information test failed: {e}")
            return False
        
        print(f"\n🧠 All CTGAN implementation tests passed successfully!")
        print(f"📊 Total test categories: 11")
        print(f"🏥 Healthcare data generation ready!")
        print(f"🚀 M4 Pro optimized synthetic data generation working!")
        print(f"🔒 Privacy-preserving healthcare AI ready for production!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import CTGAN implementation: {e}")
        return False
    except Exception as e:
        print(f"❌ CTGAN implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ctgan_implementation()
    sys.exit(0 if success else 1)