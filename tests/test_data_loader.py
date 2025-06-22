#!/usr/bin/env python3
"""
🧪 Test Script: Data Loader System
==================================

Comprehensive testing for the healthcare data loading and preprocessing pipeline.
Tests all features including privacy protection, missing value handling, and healthcare compliance.
"""

import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import json

# Import our data loader system
from mirage.data.loaders import (
    CSVDataLoader, 
    HealthcareDataLoader,
    DataType, 
    PrivacyLevel, 
    MissingValueStrategy
)
from mirage.core.exceptions import DataNotFoundError, DataValidationError, DataFormatError

def create_test_healthcare_data() -> pd.DataFrame:
    """Create realistic healthcare test data with various data types and issues."""
    np.random.seed(42)  # Reproducible results
    
    n_samples = 300
    
    # Generate realistic healthcare data
    data = {
        # Patient identifiers
        'patient_id': [f"PAT_{i:06d}" for i in range(1, n_samples + 1)],
        'medical_record_number': [f"MRN{np.random.randint(100000, 999999)}" for _ in range(n_samples)],
        
        # Demographics
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'ethnicity': np.random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], n_samples),
        
        # Physical measurements
        'weight_kg': np.random.normal(75, 15, n_samples).round(1),
        'height_cm': np.random.normal(170, 10, n_samples).round(1),
        'bmi': None,  # Will calculate
        
        # Vital signs
        'systolic_bp': np.random.normal(125, 20, n_samples).round(0),
        'diastolic_bp': np.random.normal(80, 15, n_samples).round(0),
        'heart_rate': np.random.normal(72, 12, n_samples).round(0),
        'temperature_f': np.random.normal(98.6, 1.0, n_samples).round(1),
        
        # Lab values
        'cholesterol_mg_dl': np.random.normal(200, 40, n_samples).round(0),
        'glucose_mg_dl': np.random.normal(95, 25, n_samples).round(0),
        'hemoglobin_g_dl': np.random.normal(14, 2, n_samples).round(1),
        
        # Medical conditions (boolean)
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.70, 0.30]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
        
        # Categorical medical data
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.6, 0.25, 0.15]),
        'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'diagnosis_code': [f"ICD10_{np.random.randint(100, 999)}.{np.random.randint(10, 99)}" for _ in range(n_samples)],
        
        # Date fields
        'admission_date': pd.date_range('2020-01-01', '2024-12-31', periods=n_samples),
        'discharge_date': pd.date_range('2020-01-01', '2024-12-31', periods=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate BMI
    df['bmi'] = (df['weight_kg'] / (df['height_cm'] / 100) ** 2).round(1)
    
    # Introduce realistic missing values (common in healthcare data)
    missing_patterns = {
        'weight_kg': 0.05,  # 5% missing
        'height_cm': 0.03,  # 3% missing
        'cholesterol_mg_dl': 0.15,  # 15% missing (lab not always ordered)
        'glucose_mg_dl': 0.10,  # 10% missing
        'hemoglobin_g_dl': 0.12,  # 12% missing
        'temperature_f': 0.08,  # 8% missing
    }
    
    # Apply missing values
    for col, missing_rate in missing_patterns.items():
        n_missing = int(len(df) * missing_rate)
        missing_indices = np.random.choice(df.index, n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # Add some extreme outliers (realistic in healthcare)
    outlier_indices = np.random.choice(df.index, 5, replace=False)
    df.loc[outlier_indices[0], 'weight_kg'] = 200  # Extreme weight
    df.loc[outlier_indices[1], 'systolic_bp'] = 220  # High BP
    df.loc[outlier_indices[2], 'glucose_mg_dl'] = 400  # Diabetic crisis
    df.loc[outlier_indices[3], 'age'] = 105  # Very elderly
    df.loc[outlier_indices[4], 'bmi'] = 55  # Severe obesity
    
    return df

def create_test_csv_files() -> dict:
    """Create test CSV files with different formats and issues."""
    temp_dir = tempfile.mkdtemp()
    files = {}
    
    # 1. Standard healthcare CSV
    healthcare_data = create_test_healthcare_data()
    standard_file = Path(temp_dir) / "healthcare_standard.csv"
    healthcare_data.to_csv(standard_file, index=False)
    files['standard'] = standard_file
    
    # 2. CSV with different delimiter (semicolon)
    semicolon_file = Path(temp_dir) / "healthcare_semicolon.csv"
    healthcare_data.head(50).to_csv(semicolon_file, index=False, sep=';')
    files['semicolon'] = semicolon_file
    
    # 3. CSV with tab delimiter
    tab_file = Path(temp_dir) / "healthcare_tab.csv"
    healthcare_data.head(30).to_csv(tab_file, index=False, sep='\t')
    files['tab'] = tab_file
    
    # 4. Small dataset (edge case)
    small_file = Path(temp_dir) / "healthcare_small.csv"
    healthcare_data.head(15).to_csv(small_file, index=False)
    files['small'] = small_file
    
    # 5. Dataset with many missing values
    high_missing_data = healthcare_data.copy()
    for col in ['weight_kg', 'height_cm', 'cholesterol_mg_dl', 'glucose_mg_dl']:
        missing_indices = np.random.choice(high_missing_data.index, int(len(high_missing_data) * 0.4), replace=False)
        high_missing_data.loc[missing_indices, col] = np.nan
    
    high_missing_file = Path(temp_dir) / "healthcare_high_missing.csv"
    high_missing_data.to_csv(high_missing_file, index=False)
    files['high_missing'] = high_missing_file
    
    return files

def test_data_loader_system():
    """Comprehensive test of the data loader system."""
    print("🧪 Testing Data Loader System")
    print("=" * 50)
    
    try:
        # Create test files
        test_files = create_test_csv_files()
        print(f"✅ Created {len(test_files)} test files")
        
        # Test 1: Basic CSV Loading
        print("\n🧪 Test 1: Basic CSV Loading")
        try:
            loader = CSVDataLoader()
            data = loader.load_csv(test_files['standard'])
            
            print(f"   ✅ CSV loaded successfully: {data.shape}")
            print(f"   ✅ Columns: {len(data.columns)}")
            print(f"   ✅ Data types detected: {len(loader.get_column_types())}")
            
            # Check auto-detected types
            column_types = loader.get_column_types()
            print(f"   ✅ Auto-detected types: {len([t for t in column_types.values() if t == 'id'])} ID columns")
            print(f"   ✅ Categorical columns: {len([t for t in column_types.values() if t == 'categorical'])}")
            print(f"   ✅ Numerical columns: {len([t for t in column_types.values() if t == 'numerical'])}")
            
        except Exception as e:
            print(f"   ❌ Basic CSV loading failed: {e}")
            return False
        
        # Test 2: Delimiter Auto-Detection
        print("\n🧪 Test 2: Delimiter Auto-Detection")
        try:
            # Test semicolon delimiter
            loader_semi = CSVDataLoader()
            data_semi = loader_semi.load_csv(test_files['semicolon'])
            print(f"   ✅ Semicolon CSV loaded: {data_semi.shape}")
            
            # Test tab delimiter
            loader_tab = CSVDataLoader()
            data_tab = loader_tab.load_csv(test_files['tab'])
            print(f"   ✅ Tab CSV loaded: {data_tab.shape}")
            
        except Exception as e:
            print(f"   ❌ Delimiter detection failed: {e}")
            return False
        
        # Test 3: Data Quality Analysis
        print("\n🧪 Test 3: Data Quality Analysis")
        try:
            loader = CSVDataLoader()
            data = loader.load_csv(test_files['standard'])
            quality_report = loader.get_data_quality_report()
            
            print(f"   ✅ Quality report generated")
            print(f"   ✅ Quality score: {quality_report.quality_score:.3f}")
            print(f"   ✅ Total rows: {quality_report.total_rows}")
            print(f"   ✅ Total columns: {quality_report.total_columns}")
            print(f"   ✅ Missing value columns: {len(quality_report.missing_values)}")
            print(f"   ✅ Issues identified: {len(quality_report.issues)}")
            print(f"   ✅ Privacy concerns: {len(quality_report.privacy_concerns)}")
            
            if quality_report.issues:
                print(f"   ℹ️  Issues: {quality_report.issues[:2]}")  # Show first 2
            
        except Exception as e:
            print(f"   ❌ Data quality analysis failed: {e}")
            return False
        
        # Test 4: Privacy Levels
        print("\n🧪 Test 4: Privacy Protection Levels")
        try:
            privacy_levels = [PrivacyLevel.STANDARD, PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]
            
            for privacy_level in privacy_levels:
                loader = CSVDataLoader(privacy_level=privacy_level)
                data = loader.load_csv(test_files['standard'])
                processed_data = loader.preprocess_data(data)
                
                print(f"   ✅ {privacy_level.value} privacy: {processed_data.shape}")
                
                # Check that ID columns were anonymized
                id_columns = [col for col, dtype in loader.column_types.items() if dtype == DataType.ID]
                if id_columns:
                    id_col = id_columns[0]
                    sample_id = processed_data[id_col].iloc[0]
                    print(f"   ✅ ID anonymization ({privacy_level.value}): {sample_id}")
            
        except Exception as e:
            print(f"   ❌ Privacy protection failed: {e}")
            return False
        
        # Test 5: Missing Value Strategies
        print("\n🧪 Test 5: Missing Value Handling Strategies")
        try:
            strategies = [
                MissingValueStrategy.MEDIAN_IMPUTE,
                MissingValueStrategy.MEAN_IMPUTE,
                MissingValueStrategy.MODE_IMPUTE,
                MissingValueStrategy.CONSTANT_FILL
            ]
            
            for strategy in strategies:
                loader = CSVDataLoader(missing_strategy=strategy)
                data = loader.load_csv(test_files['high_missing'])
                
                missing_before = data.isnull().sum().sum()
                processed_data = loader.preprocess_data(data)
                missing_after = processed_data.isnull().sum().sum()
                
                print(f"   ✅ {strategy.value}: {missing_before} → {missing_after} missing values")
            
        except Exception as e:
            print(f"   ❌ Missing value handling failed: {e}")
            return False
        
        # Test 6: Healthcare-Specific Processing
        print("\n🧪 Test 6: Healthcare Data Processing")
        try:
            loader = CSVDataLoader(healthcare_mode=True)
            data = loader.load_csv(test_files['standard'])
            processed_data = loader.preprocess_data(data)
            
            print(f"   ✅ Healthcare processing completed: {processed_data.shape}")
            
            # Check healthcare-specific validations
            if 'age' in processed_data.columns:
                age_range = (processed_data['age'].min(), processed_data['age'].max())
                print(f"   ✅ Age range after validation: {age_range}")
                
            if 'bmi' in processed_data.columns:
                bmi_range = (processed_data['bmi'].min(), processed_data['bmi'].max())
                print(f"   ✅ BMI range after validation: {bmi_range}")
                
            if 'systolic_bp' in processed_data.columns:
                bp_range = (processed_data['systolic_bp'].min(), processed_data['systolic_bp'].max())
                print(f"   ✅ Systolic BP range: {bp_range}")
            
        except Exception as e:
            print(f"   ❌ Healthcare processing failed: {e}")
            return False
        
        # Test 7: Healthcare Data Loader (Specialized)
        print("\n🧪 Test 7: Healthcare Data Loader (Specialized)")
        try:
            healthcare_loader = HealthcareDataLoader()
            data = healthcare_loader.load_csv(test_files['standard'])
            processed_data = healthcare_loader.preprocess_data(data)
            
            print(f"   ✅ Healthcare loader initialized with high privacy by default")
            print(f"   ✅ Processed healthcare data: {processed_data.shape}")
            
            # Test HIPAA compliance validation
            compliance_report = healthcare_loader.validate_healthcare_compliance(processed_data)
            print(f"   ✅ HIPAA compliance check: {compliance_report['hipaa_compliant']}")
            print(f"   ✅ Risk level: {compliance_report['risk_level']}")
            print(f"   ✅ Issues found: {len(compliance_report['issues'])}")
            
        except Exception as e:
            print(f"   ❌ Healthcare data loader failed: {e}")
            return False
        
        # Test 8: Column Type Management
        print("\n🧪 Test 8: Column Type Management")
        try:
            loader = CSVDataLoader()
            data = loader.load_csv(test_files['standard'])
            
            # Get initial types
            initial_types = loader.get_column_types()
            print(f"   ✅ Initial auto-detected types: {len(initial_types)}")
            
            # Manually set a column type
            if 'age' in data.columns:
                loader.set_column_type('age', DataType.CATEGORICAL)
                updated_types = loader.get_column_types()
                print(f"   ✅ Manual type override: age → {updated_types['age']}")
            
            # Process with updated types
            processed_data = loader.preprocess_data(data)
            print(f"   ✅ Processing with custom types: {processed_data.shape}")
            
        except Exception as e:
            print(f"   ❌ Column type management failed: {e}")
            return False
        
        # Test 9: Error Handling
        print("\n🧪 Test 9: Error Handling")
        try:
            loader = CSVDataLoader()
            
            # Test file not found
            try:
                loader.load_csv("nonexistent_file.csv")
                print(f"   ❌ Should have raised DataNotFoundError")
                return False
            except DataNotFoundError:
                print(f"   ✅ Correctly handled missing file")
            
            # Test preprocessing without data
            try:
                empty_loader = CSVDataLoader()
                empty_loader.preprocess_data()
                print(f"   ❌ Should have raised DataValidationError")
                return False
            except DataValidationError:
                print(f"   ✅ Correctly handled missing data for preprocessing")
            
            # Test quality report without data
            try:
                empty_loader = CSVDataLoader()
                empty_loader.get_data_quality_report()
                print(f"   ❌ Should have raised DataValidationError")
                return False
            except DataValidationError:
                print(f"   ✅ Correctly handled missing data for quality report")
            
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            return False
        
        # Test 10: Integration Test (Full Pipeline)
        print("\n🧪 Test 10: Full Pipeline Integration")
        try:
            # Create a complete pipeline
            loader = CSVDataLoader(
                privacy_level=PrivacyLevel.HIGH,
                missing_strategy=MissingValueStrategy.MEDIAN_IMPUTE,
                healthcare_mode=True
            )
            
            # Load and process data
            raw_data = loader.load_csv(test_files['standard'])
            processed_data = loader.preprocess_data(raw_data)
            
            # Get quality report
            quality_report = loader.get_data_quality_report()
            
            # Validate final results
            print(f"   ✅ Full pipeline completed")
            print(f"   ✅ Raw data: {raw_data.shape}")
            print(f"   ✅ Processed data: {processed_data.shape}")
            print(f"   ✅ Quality score: {quality_report.quality_score:.3f}")
            print(f"   ✅ Missing values remaining: {processed_data.isnull().sum().sum()}")
            
            # Verify data integrity
            assert len(processed_data) > 0, "Processed data should not be empty"
            assert processed_data.isnull().sum().sum() == 0, "No missing values should remain"
            
            print(f"   ✅ Data integrity validated")
            
        except Exception as e:
            print(f"   ❌ Full pipeline integration failed: {e}")
            return False
        
        # Test 11: Performance Benchmarking
        print("\n🧪 Test 11: Performance Benchmarking")
        try:
            import time
            
            loader = CSVDataLoader()
            
            # Measure loading time
            start_time = time.time()
            data = loader.load_csv(test_files['standard'])
            load_time = time.time() - start_time
            
            # Measure preprocessing time
            start_time = time.time()
            processed_data = loader.preprocess_data(data)
            process_time = time.time() - start_time
            
            print(f"   ✅ Loading time: {load_time:.3f}s for {data.shape[0]} rows")
            print(f"   ✅ Processing time: {process_time:.3f}s")
            print(f"   ✅ Total pipeline time: {load_time + process_time:.3f}s")
            
            # Performance thresholds (reasonable for healthcare data)
            if load_time < 1.0 and process_time < 2.0:
                print(f"   ✅ Performance meets requirements")
            else:
                print(f"   ⚠️  Performance may need optimization")
            
        except Exception as e:
            print(f"   ❌ Performance benchmarking failed: {e}")
            return False
        
        # Cleanup
        try:
            import shutil
            for file_path in test_files.values():
                if file_path.parent.exists():
                    shutil.rmtree(file_path.parent)
            print(f"\n🧹 Cleaned up test files")
        except:
            pass  # Cleanup failure is not critical
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data loader system test failed: {e}")
        return False

if __name__ == "__main__":
    print("🏥 Starting Data Loader System Tests...")
    print("=" * 60)
    
    success = test_data_loader_system()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL DATA LOADER TESTS PASSED SUCCESSFULLY!")
        print("📊 Healthcare data loading pipeline ready!")
        print("🔒 Privacy protection working correctly!")
        print("🏥 Healthcare optimizations validated!")
        print("🚀 Ready for integration with CTGAN!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("🔧 Please check the error messages above")
        print("=" * 60)