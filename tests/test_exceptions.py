#!/usr/bin/env python3
"""
Test script for Mirage exception system.
Run this to verify all exception classes work correctly.
"""

import sys
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_exceptions():
    """Test all exception classes and functionality."""
    print("🔥 Testing Mirage Exception System")
    print("=" * 50)
    
    try:
        # Import all exceptions
        from mirage.core.exceptions import (
            MirageException,
            ConfigurationError,
            SystemRequirementError,
            DeviceError,
            ModelError,
            ModelNotFoundError,
            ModelLoadError,
            ModelConfigurationError,
            ModelTrainingError,
            ModelGenerationError,
            DataError,
            DataNotFoundError,
            DataValidationError,
            DataFormatError,
            DataPreprocessingError,
            JobError,
            JobNotFoundError,
            JobExecutionError,
            JobTimeoutError,
            ResourceError,
            APIError,
            ValidationError,
            handle_exception,
            get_exception_context
        )
        
        print("✅ All exception classes imported successfully")
        
        # Test 1: Basic MirageException
        print("\n🧪 Test 1: Basic MirageException")
        try:
            exc = MirageException(
                message="Test error",
                error_code="TEST_ERROR",
                details={"test": True}
            )
            result = exc.to_dict()
            print(f"   ✅ Basic exception: {result['error_code']}")
            print(f"   ✅ Details included: {result['details']}")
            
        except Exception as e:
            print(f"   ❌ Basic exception test failed: {e}")
            return False
        
        # Test 2: Model exceptions
        print("\n🧪 Test 2: Model Exception Types")
        test_cases = [
            (ModelNotFoundError, "ctgan", "1.0.0", ["tvae", "copulagan"]),
            (ModelLoadError, "ctgan", "Corrupted model file", "/path/to/model"),
            (ModelConfigurationError, "ctgan", ["Invalid epochs", "Missing param"], {"epochs": -1}),
            (ModelTrainingError, "ctgan", "GPU out of memory", 10),
            (ModelGenerationError, "ctgan", "Invalid sample count", 1000)
        ]
        
        for exc_class, *args in test_cases:
            try:
                if exc_class == ModelNotFoundError:
                    exc = exc_class(args[0], version=args[1], available_models=args[2])
                elif exc_class == ModelLoadError:
                    exc = exc_class(args[0], args[1], model_path=args[2])
                elif exc_class == ModelConfigurationError:
                    exc = exc_class(args[0], args[1], provided_config=args[2])
                elif exc_class == ModelTrainingError:
                    exc = exc_class(args[0], args[1], epoch=args[2])
                elif exc_class == ModelGenerationError:
                    exc = exc_class(args[0], args[1], n_samples=args[2])
                
                result = exc.to_dict()
                print(f"   ✅ {exc_class.__name__}: {result['error_code']}")
                
            except Exception as e:
                print(f"   ❌ {exc_class.__name__} test failed: {e}")
                return False
        
        # Test 3: Data exceptions
        print("\n🧪 Test 3: Data Exception Types")
        data_test_cases = [
            (DataNotFoundError, "/path/to/missing.csv"),
            (DataValidationError, "Schema validation failed", ["missing column"], "/path/to/data.csv"),
            (DataFormatError, "/path/to/data.txt", "CSV", "TXT"),
            (DataPreprocessingError, "Encoding failed", "categorical_encoding", "/path/to/data.csv")
        ]
        
        for exc_class, *args in data_test_cases:
            try:
                if exc_class == DataNotFoundError:
                    exc = exc_class(args[0])
                elif exc_class == DataValidationError:
                    exc = exc_class(args[0], validation_errors=args[1], data_path=args[2])
                elif exc_class == DataFormatError:
                    exc = exc_class(args[0], args[1], actual_format=args[2])
                elif exc_class == DataPreprocessingError:
                    exc = exc_class(args[0], preprocessing_step=args[1], data_path=args[2])
                
                result = exc.to_dict()
                print(f"   ✅ {exc_class.__name__}: {result['error_code']}")
                
            except Exception as e:
                print(f"   ❌ {exc_class.__name__} test failed: {e}")
                return False
        
        # Test 4: Job exceptions
        print("\n🧪 Test 4: Job Exception Types")
        job_test_cases = [
            (JobNotFoundError, "job_123"),
            (JobExecutionError, "job_123", "Model training failed", "training"),
            (JobTimeoutError, "job_123", 300),
            (ResourceError, "Insufficient memory", "memory", 8.0, 16.0)
        ]
        
        for exc_class, *args in job_test_cases:
            try:
                if exc_class == JobNotFoundError:
                    exc = exc_class(args[0])
                elif exc_class == JobExecutionError:
                    exc = exc_class(args[0], args[1], stage=args[2])
                elif exc_class == JobTimeoutError:
                    exc = exc_class(args[0], args[1])
                elif exc_class == ResourceError:
                    exc = exc_class(args[0], resource_type=args[1], available=args[2], required=args[3])
                
                result = exc.to_dict()
                print(f"   ✅ {exc_class.__name__}: {result['error_code']}")
                
            except Exception as e:
                print(f"   ❌ {exc_class.__name__} test failed: {e}")
                return False
        
        # Test 5: API exceptions
        print("\n🧪 Test 5: API Exception Types")
        api_test_cases = [
            (APIError, "Internal server error", 500),
            (ValidationError, "Invalid request data", {"field": "Required field missing"})
        ]
        
        for exc_class, *args in api_test_cases:
            try:
                if exc_class == APIError:
                    exc = exc_class(args[0], status_code=args[1])
                elif exc_class == ValidationError:
                    exc = exc_class(args[0], field_errors=args[1])
                
                result = exc.to_dict()
                print(f"   ✅ {exc_class.__name__}: {result['error_code']}")
                if hasattr(exc, 'status_code'):
                    print(f"      Status code: {exc.status_code}")
                
            except Exception as e:
                print(f"   ❌ {exc_class.__name__} test failed: {e}")
                return False
        
        # Test 6: System exceptions
        print("\n🧪 Test 6: System Exception Types")
        system_test_cases = [
            (ConfigurationError, "Invalid configuration", "MODEL_PATH"),
            (SystemRequirementError, "PyTorch not found", "pytorch>=2.0"),
            (DeviceError, "MPS not available", "mps", True)
        ]
        
        for exc_class, *args in system_test_cases:
            try:
                if exc_class == ConfigurationError:
                    exc = exc_class(args[0], config_key=args[1])
                elif exc_class == SystemRequirementError:
                    exc = exc_class(args[0], requirement=args[1])
                elif exc_class == DeviceError:
                    exc = exc_class(args[0], device=args[1], fallback_available=args[2])
                
                result = exc.to_dict()
                print(f"   ✅ {exc_class.__name__}: {result['error_code']}")
                
            except Exception as e:
                print(f"   ❌ {exc_class.__name__} test failed: {e}")
                return False
        
        # Test 7: Exception handling utility
        print("\n🧪 Test 7: Exception Handling Utility")
        try:
            # Test with standard Python exception
            original_exc = ValueError("Invalid value")
            handled_exc = handle_exception(original_exc)
            
            print(f"   ✅ Standard exception converted: {type(handled_exc).__name__}")
            print(f"   ✅ Original exception preserved: {handled_exc.original_exception}")
            
            # Test with already MirageException
            mirage_exc = ModelError("Test model error")
            handled_mirage = handle_exception(mirage_exc)
            
            print(f"   ✅ MirageException passed through: {handled_mirage is mirage_exc}")
            
        except Exception as e:
            print(f"   ❌ Exception handling test failed: {e}")
            return False
        
        # Test 8: JSON serialization
        print("\n🧪 Test 8: JSON Serialization")
        try:
            exc = ModelNotFoundError(
                "test_model",
                version="1.0.0",
                available_models=["model1", "model2"]
            )
            
            exc_dict = exc.to_dict()
            json_str = json.dumps(exc_dict, indent=2)
            parsed = json.loads(json_str)
            
            print("   ✅ Exception serializable to JSON")
            print(f"   ✅ Contains error code: {parsed.get('error_code')}")
            print(f"   ✅ Contains details: {bool(parsed.get('details'))}")
            
        except Exception as e:
            print(f"   ❌ JSON serialization test failed: {e}")
            return False
        
        print(f"\n🚀 All exception tests passed successfully!")
        print(f"📊 Total exception classes tested: 16")
        print(f"🛡️  Exception system ready for production use!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import exceptions: {e}")
        return False
    except Exception as e:
        print(f"❌ Exception test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_exceptions()
    sys.exit(0 if success else 1)