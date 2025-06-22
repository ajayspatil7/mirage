#!/usr/bin/env python3
"""
Test script for Mirage configuration.
Run this to verify everything is working correctly.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config():
    """Test the configuration system."""
    print("🔥 Testing Mirage Configuration")
    print("=" * 50)
    
    try:
        # Import and test configuration
        from mirage.core.config import config, get_system_info
        
        print("✅ Configuration imported successfully")
        
        # Test system info
        info = get_system_info()
        
        print(f"\n🖥️  System Information:")
        print(f"   Platform: {info['system']['platform']}")
        print(f"   Python: {info['system']['python_version']}")
        print(f"   CPU Count: {info['system']['cpu_count']}")
        print(f"   Apple Silicon: {info['system']['is_apple_silicon']}")
        print(f"   M4 Pro: {info['system']['is_m4_pro']}")
        
        print(f"\n🔥 PyTorch Configuration:")
        print(f"   Version: {info['pytorch']['version']}")
        print(f"   MPS Available: {info['pytorch']['mps_available']}")
        print(f"   MPS Built: {info['pytorch']['mps_built']}")
        print(f"   Device: {info['pytorch']['device']}")
        print(f"   Threads: {info['pytorch']['thread_count']}")
        
        print(f"\n📁 Storage Configuration:")
        print(f"   Base Path: {info['storage']['base_path']}")
        print(f"   Directories Created: {info['storage']['paths_exist']}")
        
        print(f"\n⚙️  Settings:")
        print(f"   Environment: {info['settings']['environment']}")
        print(f"   Debug: {info['settings']['debug']}")
        print(f"   Max Jobs: {info['settings']['max_concurrent_jobs']}")
        
        # Test device creation
        import torch
        print(f"\n🧪 Device Test:")
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=config.device)
            result = test_tensor.sum()
            print(f"   ✅ Device {config.device} working: {result.item()}")
        except Exception as e:
            print(f"   ❌ Device test failed: {e}")
            
        # Test storage paths
        print(f"\n📂 Storage Paths:")
        for name, path in config.storage_paths.items():
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {name}: {path}")
            
        print(f"\n🚀 Configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)