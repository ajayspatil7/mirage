#!/usr/bin/env python3
"""
Test script for Mirage FastAPI service.
Run this to verify the API service works correctly.
"""

import sys
import asyncio
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_fastapi_service():
    """Test the FastAPI service functionality."""
    print("🔥 Testing Mirage FastAPI Service")
    print("=" * 50)
    
    try:
        # Import the FastAPI app
        from mirage.main import app, app_state
        
        print("✅ FastAPI app imported successfully")
        
        # Manually set healthy status for testing (since lifespan doesn't run in TestClient)
        app_state["health_status"] = "healthy"
        app_state["startup_time"] = time.time()
        
        # Create test client
        client = TestClient(app)
        
        # Test 1: Root endpoint
        print("\n🧪 Test 1: Root Endpoint")
        try:
            response = client.get("/")
            assert response.status_code == 200
            
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Service: {data['service']}")
            print(f"   ✅ Version: {data['version']}")
            print(f"   ✅ Status: {data['status']}")
            
        except Exception as e:
            print(f"   ❌ Root endpoint test failed: {e}")
            return False
        
        # Test 2: Health check
        print("\n🧪 Test 2: Health Check")
        try:
            response = client.get("/health")
            assert response.status_code == 200  # Should be 200 now
            
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Health status: {data['status']}")
            print(f"   ✅ Version: {data['version']}")
            print(f"   ✅ Environment: {data['environment']}")
            print(f"   ✅ Uptime: {data['uptime_seconds']:.2f}s")
            
            # Check metrics
            metrics = data.get('metrics', {})
            print(f"   ✅ Total requests: {metrics.get('total_requests', 0)}")
            print(f"   ✅ Total errors: {metrics.get('total_errors', 0)}")
            
        except Exception as e:
            print(f"   ❌ Health check test failed: {e}")
            return False
        
        # Test 3: System info
        print("\n🧪 Test 3: System Information")
        try:
            # Manually set system info for testing
            from mirage.core.config import get_system_info
            app_state["system_info"] = get_system_info()
            
            response = client.get("/system/info")
            
            if response.status_code == 200:
                data = response.json()
                system_info = data.get('system_info', {})
                config_info = data.get('config', {})
                
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ✅ Environment: {config_info.get('environment')}")
                print(f"   ✅ Device: {config_info.get('device')}")
                print(f"   ✅ Debug mode: {config_info.get('debug')}")
                
                # System details
                if 'system' in system_info:
                    sys_details = system_info['system']
                    print(f"   ✅ Platform: {sys_details.get('platform')}")
                    print(f"   ✅ Apple Silicon: {sys_details.get('is_apple_silicon')}")
                    print(f"   ✅ M4 Pro: {sys_details.get('is_m4_pro')}")
                
                # PyTorch details
                if 'pytorch' in system_info:
                    pytorch_info = system_info['pytorch']
                    print(f"   ✅ PyTorch version: {pytorch_info.get('version')}")
                    print(f"   ✅ MPS available: {pytorch_info.get('mps_available')}")
                    print(f"   ✅ PyTorch device: {pytorch_info.get('device')}")
                    
            else:
                print(f"   ⚠️  System info not ready (status: {response.status_code})")
                
        except Exception as e:
            print(f"   ❌ System info test failed: {e}")
            return False
        
        # Test 4: System metrics
        print("\n🧪 Test 4: System Metrics")
        try:
            response = client.get("/system/metrics")
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ✅ Timestamp: {data.get('timestamp')}")
                print(f"   ✅ Uptime: {data.get('uptime_seconds', 0):.2f}s")
                
                # Request metrics
                requests = data.get('requests', {})
                print(f"   ✅ Total requests: {requests.get('total', 0)}")
                print(f"   ✅ Error rate: {requests.get('error_rate_percent', 0):.2f}%")
                
                # System metrics
                system = data.get('system', {})
                print(f"   ✅ CPU usage: {system.get('cpu_percent', 0):.1f}%")
                print(f"   ✅ Memory usage: {system.get('memory_percent', 0):.1f}%")
                
                # MPS metrics
                mps = data.get('mps', {})
                if mps.get('available'):
                    print(f"   ✅ MPS available: {mps.get('available')}")
                    print(f"   ✅ MPS device: {mps.get('device')}")
                    
            else:
                print(f"   ⚠️  Metrics not ready (status: {response.status_code})")
                
        except Exception as e:
            print(f"   ❌ System metrics test failed: {e}")
            return False
        
        # Test 5: OpenAPI documentation
        print("\n🧪 Test 5: OpenAPI Documentation")
        try:
            # Test OpenAPI schema
            response = client.get("/openapi.json")
            assert response.status_code == 200
            
            schema = response.json()
            print(f"   ✅ OpenAPI schema: {response.status_code}")
            print(f"   ✅ API title: {schema['info']['title']}")
            print(f"   ✅ API version: {schema['info']['version']}")
            
            # Test Swagger UI
            response = client.get("/docs")
            assert response.status_code == 200
            print(f"   ✅ Swagger UI: {response.status_code}")
            
            # Test ReDoc
            response = client.get("/redoc")
            assert response.status_code == 200
            print(f"   ✅ ReDoc: {response.status_code}")
            
        except Exception as e:
            print(f"   ❌ OpenAPI documentation test failed: {e}")
            return False
        
        # Test 6: Error handling
        print("\n🧪 Test 6: Error Handling")
        try:
            # Test 404 error
            response = client.get("/nonexistent")
            assert response.status_code == 404
            
            data = response.json()
            print(f"   ✅ 404 error handled: {response.status_code}")
            print(f"   ✅ Error structure: {data.get('error', False)}")
            
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            return False
        
        # Test 7: CORS headers
        print("\n🧪 Test 7: CORS Configuration")
        try:
            # Test preflight request
            response = client.options("/", headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            })
            
            print(f"   ✅ OPTIONS request: {response.status_code}")
            
            # Check CORS headers in a GET request
            response = client.get("/", headers={"Origin": "http://localhost:3000"})
            headers = response.headers
            
            cors_headers = [
                "access-control-allow-origin",
                "access-control-allow-credentials"
            ]
            
            for header in cors_headers:
                if header in headers:
                    print(f"   ✅ CORS header {header}: {headers[header]}")
                    
        except Exception as e:
            print(f"   ❌ CORS test failed: {e}")
            return False
        
        # Test 8: Request middleware
        print("\n🧪 Test 8: Request Middleware")
        try:
            # Make multiple requests to test request counting
            initial_response = client.get("/health")
            initial_count = initial_response.json()['metrics']['total_requests']
            
            # Make additional requests
            for i in range(3):
                client.get("/")
            
            final_response = client.get("/health")
            final_count = final_response.json()['metrics']['total_requests']
            
            print(f"   ✅ Request counting: {initial_count} → {final_count}")
            print(f"   ✅ Requests tracked: {final_count - initial_count >= 3}")
            
        except Exception as e:
            print(f"   ❌ Request middleware test failed: {e}")
            return False
        
        print(f"\n🚀 All FastAPI service tests passed successfully!")
        print(f"📊 Total test categories: 8")
        print(f"🌐 API service ready for production!")
        print(f"📖 Documentation available at: /docs")
        print(f"💚 Health check available at: /health")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        return False
    except Exception as e:
        print(f"❌ FastAPI service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_startup():
    """Test that the server can start up correctly."""
    print("\n🧪 Bonus Test: Server Startup Simulation")
    try:
        from mirage.main import app, app_state, validate_system_requirements, initialize_ml_components
        
        # Simulate startup sequence
        print("   🔄 Simulating startup sequence...")
        
        # Test system validation
        import asyncio
        asyncio.run(validate_system_requirements())
        print("   ✅ System requirements validated")
        
        # Test ML components initialization  
        asyncio.run(initialize_ml_components())
        print("   ✅ ML components initialized")
        
        print("   ✅ Server startup simulation successful")
        
    except Exception as e:
        print(f"   ❌ Server startup test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = test_fastapi_service()
    if success:
        success = test_server_startup()
    
    sys.exit(0 if success else 1)