"""
StatisticalAnalyzer - Computes comprehensive statistical profiles of extracted data.

This module provides detailed statistical analysis of SEGY trace data to support
validation of attribute byte location hypotheses.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from models.data_models import StatisticalProfile


@dataclass
class DataTypeInfo:
    """Information about detected data type"""
    detected_type: str  # 'int16', 'int32', 'float32', 'float64', 'unknown'
    confidence: float  # 0.0 to 1.0
    precision: int  # Number of decimal places for floats
    has_invalid_values: bool
    byte_order: str  # 'big', 'little', 'unknown'


@dataclass
class DistributionMetrics:
    """Distribution analysis metrics"""
    skewness: float
    kurtosis: float
    percentiles: Dict[int, float]
    is_normal: bool
    normality_p_value: float
    outlier_indices: List[int]
    outlier_count: int


class StatisticalAnalyzer:
    """Computes comprehensive statistical profiles of extracted data"""
    
    def __init__(self):
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
        self.min_samples_for_analysis = 10
        
    def analyze_data(self, data: np.ndarray, attribute_type: str) -> StatisticalProfile:
        """
        Perform comprehensive statistical analysis of extracted data.
        
        Args:
            data: Numpy array of extracted values
            attribute_type: Type of attribute being analyzed (e.g., 'coordinate', 'time', 'amplitude')
            
        Returns:
            StatisticalProfile with comprehensive analysis
        """
        if len(data) == 0:
            return self._create_empty_profile()
        
        # Remove invalid values (NaN, inf) for analysis
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return self._create_empty_profile()
        
        # Basic statistics
        mean_val = float(np.mean(valid_data))
        median_val = float(np.median(valid_data))
        std_val = float(np.std(valid_data))
        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
        
        # Distribution metrics
        distribution_metrics = self.compute_distribution_metrics(valid_data)
        
        # Pattern analysis
        is_monotonic = self._check_monotonicity(valid_data)
        has_periodicity = self._detect_periodicity(valid_data)
        outlier_count = distribution_metrics.outlier_count
        null_count = len(data) - len(valid_data)
        
        # Data type analysis
        data_type_info = self.detect_data_type(data)
        
        return StatisticalProfile(
            mean=mean_val,
            median=median_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            skewness=distribution_metrics.skewness,
            kurtosis=distribution_metrics.kurtosis,
            percentiles=distribution_metrics.percentiles,
            is_monotonic=is_monotonic,
            has_periodicity=has_periodicity,
            outlier_count=outlier_count,
            null_count=null_count,
            detected_type=data_type_info.detected_type,
            precision=data_type_info.precision,
            has_invalid_values=data_type_info.has_invalid_values
        )
    
    def detect_data_type(self, data: np.ndarray) -> DataTypeInfo:
        """
        Detect the most likely data type of the extracted values.
        
        Args:
            data: Numpy array of extracted values
            
        Returns:
            DataTypeInfo with type detection results
        """
        if len(data) == 0:
            return DataTypeInfo(
                detected_type='unknown',
                confidence=0.0,
                precision=0,
                has_invalid_values=True,
                byte_order='unknown'
            )
        
        # Check for invalid values
        has_invalid = not np.all(np.isfinite(data))
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) == 0:
            return DataTypeInfo(
                detected_type='unknown',
                confidence=0.0,
                precision=0,
                has_invalid_values=True,
                byte_order='unknown'
            )
        
        # Check if all values are integers
        is_integer = np.all(valid_data == np.round(valid_data))
        
        if is_integer:
            # Determine integer size based on range
            min_val, max_val = np.min(valid_data), np.max(valid_data)
            
            if min_val >= -32768 and max_val <= 32767:
                detected_type = 'int16'
                confidence = 0.9
            elif min_val >= -2147483648 and max_val <= 2147483647:
                detected_type = 'int32'
                confidence = 0.8
            else:
                detected_type = 'int64'
                confidence = 0.7
            
            precision = 0
        else:
            # Float data - determine precision
            precision = self._estimate_float_precision(valid_data)
            
            # Check range to determine float type
            abs_max = np.max(np.abs(valid_data))
            if abs_max < 3.4e38:  # float32 range
                detected_type = 'float32'
                confidence = 0.8
            else:
                detected_type = 'float64'
                confidence = 0.9
        
        # Estimate byte order (simplified heuristic)
        byte_order = self._estimate_byte_order(valid_data)
        
        return DataTypeInfo(
            detected_type=detected_type,
            confidence=confidence,
            precision=precision,
            has_invalid_values=has_invalid,
            byte_order=byte_order
        )
    
    def compute_distribution_metrics(self, data: np.ndarray) -> DistributionMetrics:
        """
        Compute detailed distribution analysis metrics.
        
        Args:
            data: Numpy array of valid (finite) values
            
        Returns:
            DistributionMetrics with distribution analysis
        """
        if len(data) < self.min_samples_for_analysis:
            return DistributionMetrics(
                skewness=0.0,
                kurtosis=0.0,
                percentiles={},
                is_normal=False,
                normality_p_value=0.0,
                outlier_indices=[],
                outlier_count=0
            )
        
        # Skewness and kurtosis
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))
        
        # Percentiles
        percentiles = {
            5: float(np.percentile(data, 5)),
            25: float(np.percentile(data, 25)),
            50: float(np.percentile(data, 50)),  # median
            75: float(np.percentile(data, 75)),
            95: float(np.percentile(data, 95))
        }
        
        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(data) <= 5000:
            try:
                _, p_value = stats.shapiro(data)
                is_normal = p_value > 0.05
                normality_p_value = float(p_value)
            except:
                is_normal = False
                normality_p_value = 0.0
        else:
            try:
                result = stats.anderson(data, dist='norm')
                # Anderson-Darling critical values at 5% significance
                is_normal = result.statistic < result.critical_values[2]
                normality_p_value = 0.05 if is_normal else 0.01
            except:
                is_normal = False
                normality_p_value = 0.0
        
        # Outlier detection using modified Z-score
        outlier_indices = self._detect_outliers(data)
        
        return DistributionMetrics(
            skewness=skewness,
            kurtosis=kurtosis,
            percentiles=percentiles,
            is_normal=is_normal,
            normality_p_value=normality_p_value,
            outlier_indices=outlier_indices,
            outlier_count=len(outlier_indices)
        )
    
    def _create_empty_profile(self) -> StatisticalProfile:
        """Create an empty statistical profile for invalid data"""
        return StatisticalProfile(
            mean=0.0,
            median=0.0,
            std=0.0,
            min_val=0.0,
            max_val=0.0,
            skewness=0.0,
            kurtosis=0.0,
            percentiles={},
            is_monotonic=False,
            has_periodicity=False,
            outlier_count=0,
            null_count=0,
            detected_type='unknown',
            precision=0,
            has_invalid_values=True
        )
    
    def _check_monotonicity(self, data: np.ndarray) -> bool:
        """Check if data is monotonic (increasing or decreasing)"""
        if len(data) < 2:
            return False
        
        diff = np.diff(data)
        return np.all(diff >= 0) or np.all(diff <= 0)
    
    def _detect_periodicity(self, data: np.ndarray) -> bool:
        """Detect if data has periodic patterns using autocorrelation"""
        if len(data) < 20:  # Need sufficient data for periodicity detection
            return False
        
        try:
            # Compute autocorrelation
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Look for significant peaks (excluding the first one at lag 0)
            if len(autocorr) > 10:
                peaks = autocorr[2:min(len(autocorr), len(data)//4)]
                return np.max(peaks) > 0.5  # Threshold for significant periodicity
            
            return False
        except:
            return False
    
    def _detect_outliers(self, data: np.ndarray) -> List[int]:
        """Detect outliers using modified Z-score method"""
        if len(data) < 3:
            return []
        
        try:
            # Use median absolute deviation for robust outlier detection
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            
            if mad == 0:
                # If MAD is 0, use standard deviation
                std = np.std(data)
                if std == 0:
                    return []
                modified_z_scores = np.abs(data - median) / std
            else:
                modified_z_scores = 0.6745 * (data - median) / mad
            
            outlier_indices = np.where(modified_z_scores > self.outlier_threshold)[0]
            return outlier_indices.tolist()
        except:
            return []
    
    def _estimate_float_precision(self, data: np.ndarray) -> int:
        """Estimate the decimal precision of float data"""
        try:
            # Convert to strings and count decimal places
            str_data = [f"{x:.10f}".rstrip('0') for x in data[:100]]  # Sample first 100 values
            decimal_places = [len(s.split('.')[-1]) if '.' in s else 0 for s in str_data]
            
            # Return the most common precision, capped at reasonable values
            if decimal_places:
                precision = int(np.median(decimal_places))
                return min(precision, 6)  # Cap at 6 decimal places
            return 0
        except:
            return 0
    
    def _estimate_byte_order(self, data: np.ndarray) -> str:
        """Estimate byte order based on data patterns (simplified heuristic)"""
        try:
            # This is a simplified heuristic - in practice, byte order detection
            # would require knowledge of the original byte representation
            
            # Check if values seem reasonable for typical SEGY data
            abs_max = np.max(np.abs(data))
            
            # If values are extremely large, might indicate wrong byte order
            if abs_max > 1e10:
                return 'little'  # Guess opposite of typical SEGY (big-endian)
            else:
                return 'big'  # SEGY standard is big-endian
        except:
            return 'unknown'
    
    def get_attribute_specific_metrics(self, data: np.ndarray, attribute_type: str) -> Dict[str, Any]:
        """
        Get metrics specific to the attribute type being analyzed.
        
        Args:
            data: Numpy array of extracted values
            attribute_type: Type of attribute ('coordinate', 'time', 'amplitude', 'index')
            
        Returns:
            Dictionary of attribute-specific metrics
        """
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return {}
        
        metrics = {}
        
        if attribute_type.lower() in ['coordinate', 'x', 'y', 'z']:
            # Coordinate-specific metrics
            metrics['coordinate_range'] = float(np.max(valid_data) - np.min(valid_data))
            metrics['likely_utm'] = self._is_likely_utm_coordinate(valid_data)
            metrics['likely_geographic'] = self._is_likely_geographic_coordinate(valid_data)
            
        elif attribute_type.lower() in ['time', 'delay', 'sample']:
            # Time-specific metrics
            metrics['time_range_ms'] = float(np.max(valid_data) - np.min(valid_data))
            metrics['likely_sample_rate'] = self._estimate_sample_rate(valid_data)
            
        elif attribute_type.lower() in ['amplitude', 'trace']:
            # Amplitude-specific metrics
            metrics['dynamic_range'] = float(np.max(valid_data) / np.min(valid_data)) if np.min(valid_data) != 0 else float('inf')
            metrics['zero_crossings'] = self._count_zero_crossings(valid_data)
            
        elif attribute_type.lower() in ['index', 'inline', 'crossline', 'cdp']:
            # Index-specific metrics
            metrics['is_sequential'] = self._is_sequential(valid_data)
            metrics['step_size'] = self._estimate_step_size(valid_data)
            metrics['has_gaps'] = self._has_gaps(valid_data)
        
        return metrics
    
    def _is_likely_utm_coordinate(self, data: np.ndarray) -> bool:
        """Check if data looks like UTM coordinates"""
        # UTM coordinates are typically 6-7 digits
        abs_data = np.abs(data)
        return np.all((abs_data >= 100000) & (abs_data <= 9999999))
    
    def _is_likely_geographic_coordinate(self, data: np.ndarray) -> bool:
        """Check if data looks like geographic coordinates"""
        # Geographic coordinates are typically -180 to 180 for longitude, -90 to 90 for latitude
        return np.all((data >= -180) & (data <= 180))
    
    def _estimate_sample_rate(self, data: np.ndarray) -> Optional[float]:
        """Estimate sample rate from time data"""
        if len(data) < 2:
            return None
        
        try:
            # Look for common sample intervals
            diffs = np.diff(np.sort(data))
            diffs = diffs[diffs > 0]  # Remove zero differences
            
            if len(diffs) > 0:
                # Most common difference might be the sample interval
                common_diff = float(np.median(diffs))
                return common_diff
            return None
        except:
            return None
    
    def _count_zero_crossings(self, data: np.ndarray) -> int:
        """Count zero crossings in amplitude data"""
        try:
            return int(np.sum(np.diff(np.sign(data)) != 0))
        except:
            return 0
    
    def _is_sequential(self, data: np.ndarray) -> bool:
        """Check if index data is sequential"""
        if len(data) < 2:
            return False
        
        sorted_data = np.sort(data)
        diffs = np.diff(sorted_data)
        
        # Check if differences are consistent (allowing for some variation)
        if len(diffs) > 0:
            median_diff = np.median(diffs)
            return np.all(np.abs(diffs - median_diff) <= 1)
        
        return False
    
    def _estimate_step_size(self, data: np.ndarray) -> Optional[float]:
        """Estimate step size for index data"""
        if len(data) < 2:
            return None
        
        try:
            sorted_data = np.sort(data)
            diffs = np.diff(sorted_data)
            diffs = diffs[diffs > 0]
            
            if len(diffs) > 0:
                return float(np.median(diffs))
            return None
        except:
            return None
    
    def _has_gaps(self, data: np.ndarray) -> bool:
        """Check if index data has gaps"""
        if len(data) < 3:
            return False
        
        try:
            sorted_data = np.sort(data)
            diffs = np.diff(sorted_data)
            median_diff = np.median(diffs)
            
            # Look for differences significantly larger than median
            large_gaps = diffs > median_diff * 2
            return np.any(large_gaps)
        except:
            return False