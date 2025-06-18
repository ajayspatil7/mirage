"""
🏥 Syntheta Mirage - Healthcare Data Loader System
==================================================

Production-grade data loading and preprocessing for synthetic data generation.
Optimized for healthcare data with privacy-preserving transformations.

Author: Syntheta Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import logging
from datetime import datetime, date
import re

from ..core.exceptions import (
    DataNotFoundError,
    DataValidationError,
    DataFormatError,
    DataPreprocessingError
)
from ..core.config import config

# Configure logger
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Supported data types for column classification"""
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    DATETIME = "datetime"
    ID = "id"
    BOOLEAN = "boolean"
    TEXT = "text"

class PrivacyLevel(Enum):
    """Privacy levels for healthcare data processing"""
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

class MissingValueStrategy(Enum):
    """Strategies for handling missing values in healthcare data"""
    DROP_ROWS = "drop_rows"
    MEAN_IMPUTE = "mean_impute"
    MEDIAN_IMPUTE = "median_impute"
    MODE_IMPUTE = "mode_impute"
    FORWARD_FILL = "forward_fill"
    CONSTANT_FILL = "constant_fill"
    KNN_IMPUTE = "knn_impute"

class DataQualityReport:
    """Comprehensive data quality assessment report"""
    
    def __init__(self):
        self.total_rows: int = 0
        self.total_columns: int = 0
        self.missing_values: Dict[str, int] = {}
        self.data_types: Dict[str, str] = {}
        self.quality_score: float = 0.0
        self.issues: List[str] = []
        self.recommendations: List[str] = []
        self.privacy_concerns: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format"""
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values": self.missing_values,
            "data_types": self.data_types,
            "quality_score": self.quality_score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "privacy_concerns": self.privacy_concerns
        }

class CSVDataLoader:
    """
    Production-grade CSV data loader optimized for healthcare data.
    
    Features:
    - Automatic schema detection and validation
    - Healthcare-specific preprocessing
    - Privacy-aware data handling
    - Comprehensive data quality reporting
    - Missing value handling strategies
    """
    
    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.STANDARD,
        missing_strategy: MissingValueStrategy = MissingValueStrategy.MEDIAN_IMPUTE,
        auto_detect_types: bool = True,
        healthcare_mode: bool = True
    ):
        """
        Initialize the CSV data loader.
        
        Args:
            privacy_level: Level of privacy protection to apply
            missing_strategy: Strategy for handling missing values
            auto_detect_types: Whether to automatically detect column types
            healthcare_mode: Enable healthcare-specific optimizations
        """
        self.privacy_level = privacy_level
        self.missing_strategy = missing_strategy
        self.auto_detect_types = auto_detect_types
        self.healthcare_mode = healthcare_mode
        
        # Configuration
        self.config = config
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, DataType] = {}
        self.quality_report: Optional[DataQualityReport] = None
        
        # Healthcare patterns for automatic detection
        self.healthcare_patterns = {
            'id_patterns': [
                r'.*id$', r'.*_id$', r'^id_.*', r'patient.*id', r'record.*id',
                r'mrn', r'medical.*record', r'chart.*number'
            ],
            'age_patterns': [
                r'^age$', r'.*age.*', r'years.*old', r'date.*birth', r'dob'
            ],
            'measurement_patterns': [
                r'.*weight.*', r'.*height.*', r'.*bmi.*', r'.*bp.*', r'.*pressure.*',
                r'.*rate.*', r'.*level.*', r'.*count.*', r'.*score.*'
            ],
            'categorical_patterns': [
                r'.*gender.*', r'.*sex.*', r'.*type.*', r'.*status.*', r'.*category.*',
                r'.*group.*', r'.*class.*', r'.*diagnosis.*'
            ]
        }
        
        logger.info(f"🏥 CSV Data Loader initialized")
        logger.info(f"🔒 Privacy level: {privacy_level.value}")
        logger.info(f"🛠️  Missing strategy: {missing_strategy.value}")
        logger.info(f"🏥 Healthcare mode: {healthcare_mode}")
    
    def load_csv(
        self,
        file_path: Union[str, Path],
        delimiter: str = None,
        encoding: str = 'utf-8',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file with automatic format detection.
        
        Args:
            file_path: Path to the CSV file
            delimiter: CSV delimiter (auto-detected if None)
            encoding: File encoding
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataNotFoundError: If file doesn't exist
            DataFormatError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"📁 Loading CSV file: {file_path}")
        
        try:
            # Auto-detect delimiter if not provided
            if delimiter is None:
                delimiter = self._detect_delimiter(file_path)
                logger.debug(f"🔍 Auto-detected delimiter: '{delimiter}'")
            
            # Load the CSV with error handling
            self.raw_data = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                **kwargs
            )
            
            logger.info(f"✅ CSV loaded successfully: {self.raw_data.shape}")
            logger.info(f"📊 Columns: {list(self.raw_data.columns)}")
            
            # Perform initial data analysis
            self._analyze_data_quality()
            
            # Auto-detect column types if enabled
            if self.auto_detect_types:
                self._detect_column_types()
            
            return self.raw_data.copy()
            
        except Exception as e:
            raise DataFormatError(f"Failed to load CSV file {file_path}: {str(e)}")
    
    def preprocess_data(
        self,
        data: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing to the data.
        
        Args:
            data: DataFrame to process (uses loaded data if None)
            target_column: Target column for supervised learning
            
        Returns:
            Preprocessed DataFrame
            
        Raises:
            DataValidationError: If data validation fails
            DataPreprocessingError: If preprocessing fails
        """
        if data is None:
            if self.raw_data is None:
                raise DataValidationError("No data loaded. Call load_csv() first.")
            data = self.raw_data.copy()
        else:
            data = data.copy()
        
        logger.info(f"🔄 Starting data preprocessing...")
        logger.info(f"📊 Input data shape: {data.shape}")
        
        try:
            # Step 1: Handle missing values
            data = self._handle_missing_values(data)
            
            # Step 2: Apply healthcare-specific preprocessing
            if self.healthcare_mode:
                data = self._apply_healthcare_preprocessing(data)
            
            # Step 3: Apply privacy protections
            data = self._apply_privacy_protection(data)
            
            # Step 4: Data type conversions
            data = self._convert_data_types(data)
            
            # Step 5: Validate processed data
            self._validate_processed_data(data)
            
            self.processed_data = data
            
            logger.info(f"✅ Preprocessing completed: {data.shape}")
            logger.info(f"🔄 Processing pipeline: missing values → healthcare → privacy → types → validation")
            
            return data
            
        except Exception as e:
            raise DataPreprocessingError(f"Data preprocessing failed: {str(e)}")
    
    def get_data_quality_report(self) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Data quality assessment report
        """
        if self.quality_report is None:
            if self.raw_data is not None:
                self._analyze_data_quality()
            else:
                raise DataValidationError("No data loaded. Call load_csv() first.")
        
        return self.quality_report
    
    def get_column_types(self) -> Dict[str, str]:
        """Get detected column types."""
        return {col: dtype.value for col, dtype in self.column_types.items()}
    
    def set_column_type(self, column: str, data_type: DataType) -> None:
        """Manually set column type."""
        if column not in self.raw_data.columns:
            raise DataValidationError(f"Column '{column}' not found in data")
        
        self.column_types[column] = data_type
        logger.debug(f"🏷️  Set {column} as {data_type.value}")
    
    def _detect_delimiter(self, file_path: Path) -> str:
        """Auto-detect CSV delimiter."""
        common_delimiters = [',', ';', '\t', '|']
        
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        delimiter_counts = {delim: first_line.count(delim) for delim in common_delimiters}
        detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        
        if delimiter_counts[detected_delimiter] == 0:
            detected_delimiter = ','  # Default fallback
        
        return detected_delimiter
    
    def _analyze_data_quality(self) -> None:
        """Analyze data quality and generate report."""
        if self.raw_data is None:
            return
        
        report = DataQualityReport()
        data = self.raw_data
        
        # Basic statistics
        report.total_rows = len(data)
        report.total_columns = len(data.columns)
        
        # Missing values analysis
        missing_counts = data.isnull().sum()
        report.missing_values = {col: int(count) for col, count in missing_counts.items() if count > 0}
        
        # Data types
        report.data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Quality score calculation
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        report.quality_score = max(0.0, 1.0 - missing_ratio)
        
        # Identify issues
        if missing_ratio > 0.3:
            report.issues.append("High missing value ratio (>30%)")
            report.recommendations.append("Consider data collection improvement")
        
        if len(data) < 100:
            report.issues.append("Small dataset size (<100 rows)")
            report.recommendations.append("Consider gathering more data for better model performance")
        
        # Privacy concerns (healthcare mode)
        if self.healthcare_mode:
            for col in data.columns:
                if any(re.search(pattern, col.lower()) for pattern in self.healthcare_patterns['id_patterns']):
                    report.privacy_concerns.append(f"Potential ID column: {col}")
        
        self.quality_report = report
        logger.info(f"📊 Data quality score: {report.quality_score:.2f}")
        
        if report.issues:
            logger.warning(f"⚠️  Data quality issues: {len(report.issues)}")
            for issue in report.issues:
                logger.warning(f"   - {issue}")
    
    def _detect_column_types(self) -> None:
        """Automatically detect column types based on content and patterns."""
        if self.raw_data is None:
            return
        
        data = self.raw_data
        
        for column in data.columns:
            col_lower = column.lower()
            
            # Check for ID patterns
            if any(re.search(pattern, col_lower) for pattern in self.healthcare_patterns['id_patterns']):
                self.column_types[column] = DataType.ID
            
            # Check for categorical patterns or low cardinality
            elif (any(re.search(pattern, col_lower) for pattern in self.healthcare_patterns['categorical_patterns']) or
                  (data[column].dtype == 'object' and data[column].nunique() < 20)):
                self.column_types[column] = DataType.CATEGORICAL
            
            # Check for numerical data
            elif pd.api.types.is_numeric_dtype(data[column]):
                self.column_types[column] = DataType.NUMERICAL
            
            # Check for datetime
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                self.column_types[column] = DataType.DATETIME
            
            # Check for boolean
            elif data[column].dtype == 'bool' or set(data[column].dropna().unique()) <= {0, 1, True, False}:
                self.column_types[column] = DataType.BOOLEAN
            
            # Default to text for object types
            elif data[column].dtype == 'object':
                self.column_types[column] = DataType.TEXT
            
            else:
                self.column_types[column] = DataType.NUMERICAL  # Default fallback
        
        logger.info(f"🔍 Auto-detected column types:")
        for col, dtype in self.column_types.items():
            logger.debug(f"   {col}: {dtype.value}")
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value handling strategy."""
        missing_before = data.isnull().sum().sum()
        
        if missing_before == 0:
            logger.info("✅ No missing values found")
            return data
        
        logger.info(f"🔄 Handling {missing_before} missing values using {self.missing_strategy.value}")
        
        if self.missing_strategy == MissingValueStrategy.DROP_ROWS:
            data = data.dropna()
        
        elif self.missing_strategy == MissingValueStrategy.MEAN_IMPUTE:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            data = data.fillna(data.mode().iloc[0])  # Mode for non-numeric
        
        elif self.missing_strategy == MissingValueStrategy.MEDIAN_IMPUTE:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            data = data.fillna(data.mode().iloc[0])  # Mode for non-numeric
        
        elif self.missing_strategy == MissingValueStrategy.MODE_IMPUTE:
            for col in data.columns:
                data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 0)
        
        elif self.missing_strategy == MissingValueStrategy.FORWARD_FILL:
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        elif self.missing_strategy == MissingValueStrategy.CONSTANT_FILL:
            data = data.fillna(0)  # Fill with 0 for all types
        
        missing_after = data.isnull().sum().sum()
        logger.info(f"✅ Missing values reduced: {missing_before} → {missing_after}")
        
        return data
    
    def _apply_healthcare_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply healthcare-specific preprocessing."""
        logger.info("🏥 Applying healthcare preprocessing...")
        
        # Ensure realistic value ranges for common healthcare metrics
        for col in data.columns:
            col_lower = col.lower()
            
            # Age validation
            if 'age' in col_lower and pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].clip(0, 120)  # Realistic age range
                logger.debug(f"🔄 Applied age validation to {col}")
            
            # BMI validation
            elif 'bmi' in col_lower and pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].clip(10, 60)  # Realistic BMI range
                logger.debug(f"🔄 Applied BMI validation to {col}")
            
            # Blood pressure validation
            elif ('systolic' in col_lower or 'diastolic' in col_lower) and pd.api.types.is_numeric_dtype(data[col]):
                if 'systolic' in col_lower:
                    data[col] = data[col].clip(70, 250)
                else:
                    data[col] = data[col].clip(40, 150)
                logger.debug(f"🔄 Applied BP validation to {col}")
            
            # Heart rate validation
            elif ('heart' in col_lower and 'rate' in col_lower) and pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].clip(30, 200)
                logger.debug(f"🔄 Applied heart rate validation to {col}")
        
        return data
    
    def _apply_privacy_protection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply privacy protection based on privacy level."""
        logger.info(f"🔒 Applying {self.privacy_level.value} privacy protection...")
        
        if self.privacy_level == PrivacyLevel.STANDARD:
            # Basic anonymization - hash ID columns
            for col, dtype in self.column_types.items():
                if dtype == DataType.ID and col in data.columns:
                    data[col] = data[col].astype(str).apply(lambda x: f"anon_{hash(x) % 10000:04d}")
                    logger.debug(f"🔒 Anonymized ID column: {col}")
        
        elif self.privacy_level == PrivacyLevel.HIGH:
            # Stronger anonymization + noise injection
            for col, dtype in self.column_types.items():
                if dtype == DataType.ID and col in data.columns:
                    data[col] = data[col].astype(str).apply(lambda x: f"secure_{hash(x + 'salt') % 100000:05d}")
                    logger.debug(f"🔒 Secured ID column: {col}")
                
                # Add small noise to numerical columns
                elif dtype == DataType.NUMERICAL and col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        noise_std = data[col].std() * 0.01  # 1% noise
                        noise = np.random.normal(0, noise_std, len(data))
                        data[col] = data[col] + noise
                        logger.debug(f"🔒 Added privacy noise to {col}")
        
        elif self.privacy_level == PrivacyLevel.MAXIMUM:
            # Maximum privacy - remove direct identifiers and add significant noise
            for col, dtype in self.column_types.items():
                if dtype == DataType.ID and col in data.columns:
                    # Replace with completely random IDs
                    data[col] = [f"max_private_{i:06d}" for i in range(len(data))]
                    logger.debug(f"🔒 Maximum privacy applied to {col}")
                
                # Add more noise to numerical columns
                elif dtype == DataType.NUMERICAL and col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        noise_std = data[col].std() * 0.03  # 3% noise
                        noise = np.random.normal(0, noise_std, len(data))
                        data[col] = data[col] + noise
                        logger.debug(f"🔒 Maximum privacy noise added to {col}")
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types based on detected column types."""
        logger.info("🔄 Converting data types...")
        
        for col, dtype in self.column_types.items():
            if col not in data.columns:
                continue
            
            try:
                if dtype == DataType.CATEGORICAL:
                    # Convert to object dtype for SDV compatibility (not category)
                    data[col] = data[col].astype('object')
                    logger.debug(f"🏷️  Converted {col} to object (SDV compatible)")
                
                elif dtype == DataType.NUMERICAL:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    logger.debug(f"🔢 Converted {col} to numerical")
                
                elif dtype == DataType.BOOLEAN:
                    data[col] = data[col].astype(bool)
                    logger.debug(f"✅ Converted {col} to boolean")
                
                elif dtype == DataType.DATETIME:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    logger.debug(f"📅 Converted {col} to datetime")
                
                # ID and TEXT columns remain as strings
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to convert {col} to {dtype.value}: {e}")
        
        return data
    
    def _validate_processed_data(self, data: pd.DataFrame) -> None:
        """Validate the processed data quality."""
        logger.info("✅ Validating processed data...")
        
        # Check for empty data
        if len(data) == 0:
            raise DataValidationError("Processed data is empty")
        
        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if np.isinf(data[numeric_cols]).any().any():
            logger.warning("⚠️  Infinite values detected in processed data")
        
        # Check data quality
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > 0.5:
            raise DataValidationError(f"Too many missing values after processing: {missing_ratio:.2%}")
        
        logger.info(f"✅ Data validation passed: {data.shape}")
        logger.info(f"📊 Final missing value ratio: {missing_ratio:.2%}")


class HealthcareDataLoader(CSVDataLoader):
    """
    Specialized data loader optimized for healthcare datasets.
    
    Includes additional healthcare-specific validations and preprocessing.
    """
    
    def __init__(self, **kwargs):
        # Force healthcare mode and appropriate defaults
        kwargs.update({
            'healthcare_mode': True,
            'privacy_level': PrivacyLevel.HIGH,
            'missing_strategy': MissingValueStrategy.MEDIAN_IMPUTE
        })
        super().__init__(**kwargs)
        
        logger.info("🏥 Healthcare Data Loader initialized with enhanced privacy settings")
    
    def validate_healthcare_compliance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data for healthcare compliance requirements.
        
        Returns:
            Compliance validation report
        """
        compliance_report = {
            "hipaa_compliant": True,
            "issues": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        # Check for direct identifiers
        potential_phi = []
        for col in data.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['name', 'ssn', 'phone', 'email', 'address']):
                potential_phi.append(col)
        
        if potential_phi:
            compliance_report["hipaa_compliant"] = False
            compliance_report["issues"].append(f"Potential PHI columns detected: {potential_phi}")
            compliance_report["recommendations"].append("Remove or de-identify PHI columns")
            compliance_report["risk_level"] = "high"
        
        # Check date precision (dates should be generalized)
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            compliance_report["recommendations"].append("Consider generalizing dates to reduce re-identification risk")
        
        logger.info(f"🔒 HIPAA compliance check: {compliance_report['hipaa_compliant']}")
        return compliance_report