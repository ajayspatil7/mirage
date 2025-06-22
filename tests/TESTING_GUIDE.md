# 🎯 Syntheta Mirage E2E Testing Framework
## Comprehensive Usage Guide

### 📋 Prerequisites

1. **Python 3.8+** (recommended: 3.10 or 3.12)
2. **Poetry** (if using poetry) or **pip**
3. **Required packages** (see installation below)

### 🚀 Quick Start

#### Method 1: Direct Python Execution
```bash
# Navigate to the mirage directory
cd /Users/ajaysp/Develop/Projects/Syntheta/mirage

# Install minimal dependencies (if not already installed)
pip install pandas numpy fastapi httpx psutil torch

# Run the comprehensive E2E test suite
python testing_e2e.py
```

#### Method 2: Using Poetry (if available)
```bash
# Navigate to the mirage directory
cd /Users/ajaysp/Develop/Projects/Syntheta/mirage

# Install dependencies
poetry install

# Run with poetry
poetry run python testing_e2e.py
```

#### Method 3: Install Optional Dependencies for Full Features
```bash
# Install all dependencies including optional ones
pip install -r requirements_test.txt

# Run the tests
python testing_e2e.py
```

### 📊 What the Tests Do

The E2E testing framework runs **6 major test categories**:

1. **Healthcare Data Pipeline** (10 tests)
   - Standard data processing validation
   - Edge case and anomaly handling
   - PHI protection and HIPAA compliance
   - Missing value strategies
   - Data type inference
   - Medical range validation

2. **API Integration** (10 tests)
   - Health check endpoints
   - Job submission and tracking
   - File upload capabilities
   - Concurrent job handling
   - Error response validation
   - API documentation checks

3. **Performance Validation** (8 tests)
   - Training performance benchmarks
   - Generation speed testing
   - Memory efficiency analysis
   - M4 Pro optimization detection
   - Resource utilization monitoring

4. **Privacy Compliance** (8 tests)
   - PHI detection with regex patterns
   - Anonymization effectiveness
   - K-anonymity validation
   - Re-identification risk assessment
   - Differential privacy mechanisms
   - Data leakage prevention

5. **Error Handling** (12 tests)
   - Invalid configuration handling
   - Corrupted data recovery
   - Memory exhaustion simulation
   - Network interruption testing
   - Timeout handling
   - Model persistence errors

6. **Production Readiness** (5 tests)
   - System monitoring capabilities
   - Logging system validation
   - Documentation completeness
   - Deployment configuration
   - Smoke testing

### 📈 Test Output

The framework generates:

1. **Real-time console output** with progress indicators:
   ```
   🎯 Starting Comprehensive E2E Testing Pipeline
   ================================================================================
   
   📅 Date: 2025-06-19 15:30:45
   🖥️  Platform: darwin
   🐍 Python: 3.12.0
   ⚡ Device: cpu
   🏥 Mode: Healthcare Data Generation
   
   📊 Test Category: Healthcare Data Pipeline
      🧪 Standard Data Processing... ✅
      🧪 Edge Case Handling... ✅
      🧪 PHI Protection... ✅
   ```

2. **Detailed JSON report** (`results/test_results.json`):
   ```json
   {
     "timestamp": "2025-06-19T15:30:45",
     "environment": {
       "platform": "darwin",
       "python_version": "3.12.0",
       "device": "cpu"
     },
     "tests": {...},
     "summary": {
       "total_tests": 53,
       "passed": 45,
       "failed": 2,
       "warnings": 6
     }
   }
   ```

3. **Comprehensive Markdown report** (`results/test_report.md`)

### 🔧 Configuration Options

You can customize the testing by modifying `TestConfig` in the file:

```python
class TestConfig:
    # Test data sizes
    SAMPLE_SIZES = [100, 1000, 5000]
    
    # Performance thresholds
    PERFORMANCE_TARGETS = {
        "training_time_per_1k_rows": 120,  # seconds
        "generation_time_per_1k_samples": 30,  # seconds
        "memory_efficiency": 0.90,
        "api_response_time": 0.5,
    }
    
    # Privacy compliance targets
    COMPLIANCE_TARGETS = {
        "k_anonymity": 5,
        "reidentification_risk": 0.05,
    }
```

### 🏥 Healthcare Data Testing

The framework includes specialized healthcare data generation:

- **Realistic correlations** (age vs weight, BMI vs blood pressure)
- **Medical range validation** (vital signs, lab values)
- **PHI simulation** (names, SSNs, phone numbers, emails)
- **Disease probability modeling** (diabetes, hypertension)
- **HIPAA compliance checking**

### 🔒 Privacy & Security Features

- **PHI Detection**: Regex patterns for SSN, phone, email
- **Anonymization**: Multi-level privacy processing
- **Differential Privacy**: Laplace noise mechanism
- **K-Anonymity**: Quasi-identifier grouping
- **Re-identification Risk**: Distance-based assessment

### 🚨 Troubleshooting

#### Common Issues:

1. **Import Errors**: 
   - The framework uses mock implementations if real Mirage components aren't available
   - Install missing dependencies: `pip install pandas numpy fastapi httpx psutil`

2. **Permission Errors**:
   - Ensure write permissions for test output directory
   - Check disk space availability

3. **Memory Issues**:
   - Reduce `SAMPLE_SIZES` in `TestConfig` for lower memory usage
   - Close other applications if running on limited memory

4. **Timeout Issues**:
   - Increase timeout values in test configurations
   - Check system performance and available resources

#### Mock Mode:
If the real Mirage engine isn't available, the framework runs in **Mock Mode**:
- Uses synthetic FastAPI endpoints
- Simulates model training and generation
- Provides realistic test scenarios
- Still validates framework functionality

### 📊 Performance Benchmarks

The framework tests against these targets:
- **Training**: <120 seconds per 1K rows
- **Generation**: <30 seconds per 1K samples
- **Memory**: >90% efficiency
- **API Response**: <0.5 seconds
- **Throughput**: >100 samples/second

### 🎯 Success Criteria

A successful test run should show:
- ✅ **80%+ tests passing**
- ✅ **HIPAA compliance validated**
- ✅ **Performance targets met**
- ✅ **No critical security issues**
- ✅ **Documentation available**

### 📞 Support

For issues or questions:
1. Check the generated test reports in `results/`
2. Review console output for specific error messages
3. Verify all dependencies are installed
4. Ensure sufficient system resources

---

**Framework Version**: 2.0.0  
**Last Updated**: June 19, 2025  
**Compatibility**: Python 3.8+, FastAPI 0.68+  
**License**: Syntheta Internal Use
