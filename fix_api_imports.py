#!/usr/bin/env python3
"""
🔧 API Import Fix Script
=======================

Simple script to ensure all API imports work correctly.
Run this before running the API integration test.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check if all required imports work"""
    print("🔍 Checking API imports...")
    
    try:
        # Core imports
        from mirage.core.config import config
        print("   ✅ Core config imported")
        
        from mirage.core.exceptions import MirageException
        print("   ✅ Core exceptions imported")
        
        # Engine imports
        from mirage.engine.serenity_engine import SerenityEngine, GenerationJob, JobPriority
        print("   ✅ SerenityEngine imported")
        
        from mirage.engine.queue import JobQueue, JobStatus
        print("   ✅ JobQueue imported")
        
        # Model imports
        from mirage.models.tabular.ctgan import CTGANGenerator, CTGANConfig
        print("   ✅ CTGAN imported")
        
        # Data loader imports
        from mirage.data.loaders import CSVDataLoader, HealthcareDataLoader
        print("   ✅ Data loaders imported")
        
        # API schema imports
        from mirage.api.schemas.requests import GenerateRequest, ModelType, DataType
        print("   ✅ Request schemas imported")
        
        from mirage.api.schemas.responses import JobSubmissionResponse, JobStatusResponse
        print("   ✅ Response schemas imported")
        
        # API endpoint imports
        from mirage.api.v1.generate import router
        print("   ✅ API router imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def check_api_components():
    """Check if API components can be instantiated"""
    print("\n🧪 Checking API component instantiation...")
    
    try:
        # Test SerenityEngine creation
        from mirage.engine.serenity_engine import SerenityEngine
        engine = SerenityEngine()
        print("   ✅ SerenityEngine created successfully")
        
        # Test CTGAN config creation
        from mirage.models.tabular.ctgan import CTGANConfig
        from mirage.api.schemas.requests import DataType
        config = CTGANConfig(
            name="test_ctgan",
            data_type=DataType.TABULAR,
            epochs=10, 
            batch_size=50
        )
        print("   ✅ CTGANConfig created successfully")
        
        # Test data loader creation
        from mirage.data.loaders import HealthcareDataLoader
        loader = HealthcareDataLoader()
        print("   ✅ HealthcareDataLoader created successfully")
        
        # Test request schema creation
        from mirage.api.schemas.requests import GenerateRequest, ModelType, DataType, PrivacyLevel, OutputFormat
        request = GenerateRequest(
            model_type=ModelType.CTGAN,
            data_type=DataType.TABULAR,
            n_samples=10,
            privacy_level=PrivacyLevel.HIGH,
            output_format=OutputFormat.CSV
        )
        print("   ✅ GenerateRequest created successfully")
        
        print("\n✅ All components instantiated successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Component instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 API Import Fix Script")
    print("=" * 40)
    
    imports_ok = check_imports()
    components_ok = check_api_components()
    
    if imports_ok and components_ok:
        print("\n🎉 API is ready for testing!")
        print("You can now run: poetry run python test_api_integration.py")
    else:
        print("\n❌ API setup issues detected")
        print("Please check the error messages above")
    
    sys.exit(0 if (imports_ok and components_ok) else 1)