"""
TraceDataValidator - Validates byte location hypotheses against actual trace data.

This module implements the core validation loop that extracts sample data from
hypothesized byte locations and validates them using statistical analysis and LLM evaluation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from models.data_models import AttributeHypothesis, ValidationResult, StatisticalProfile, ValidationStatus
from core.segy_file_handler import SEGYFileHandler
from core.statistical_analyzer import StatisticalAnalyzer
from core.validation_llm import ValidationLLM, ValidationDecision
from core.llm_provider import LLMFactory


class TraceDataValidator:
    """Validates byte location hypotheses against actual trace data"""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.statistical_analyzer = StatisticalAnalyzer()
        self.validation_llm = ValidationLLM(llm_factory)
        
        # Validation parameters
        self.default_sample_size = 100  # Number of traces to sample for validation
        self.min_sample_size = 10       # Minimum traces needed for validation
        self.max_sample_size = 500      # Maximum traces to avoid performance issues
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.5
        
    def validate_hypothesis(self, hypothesis: AttributeHypothesis, segy_handler: SEGYFileHandler) -> ValidationResult:
        """
        Core validation method that tests a hypothesis against actual trace data.
        
        Args:
            hypothesis: The attribute hypothesis to validate
            segy_handler: SEGY file handler for data extraction
            
        Returns:
            ValidationResult with validation outcome and detailed analysis
        """
        if not segy_handler.segy_file or not segy_handler.file_info:
            return self._create_failed_validation("SEGY file not properly opened")
        
        try:
            # Extract sample data from the hypothesized byte range
            sample_data = self.extract_sample_data(
                segy_handler, 
                (hypothesis.byte_start, hypothesis.byte_end),
                self.default_sample_size
            )
            
            if len(sample_data) == 0:
                return self._create_failed_validation("No data could be extracted from specified byte range")
            
            # Compute statistical profile
            statistical_profile = self.compute_statistics(sample_data, hypothesis.attribute_name)
            
            # Use LLM to evaluate if statistics match expected attribute behavior
            validation_decision = self.validation_llm.evaluate_statistics(
                statistical_profile, 
                hypothesis.attribute_name,
                context=f"SEGY file: {segy_handler.filepath}, Byte range: {hypothesis.byte_start}-{hypothesis.byte_end}"
            )
            
            # Generate refinement suggestions if validation failed
            suggestions = []
            if not validation_decision.is_valid:
                suggestions = self.validation_llm.suggest_refinements(
                    validation_decision, 
                    statistical_profile, 
                    hypothesis.attribute_name
                )
            
            # Create validation result
            return ValidationResult(
                passed=validation_decision.is_valid,
                confidence=validation_decision.confidence,
                statistical_profile=statistical_profile,
                llm_evaluation=validation_decision.reasoning,
                issues=validation_decision.issues,
                suggestions=suggestions
            )
            
        except Exception as e:
            return self._create_failed_validation(f"Validation error: {str(e)}")
    
    def extract_sample_data(self, segy_handler: SEGYFileHandler, byte_range: Tuple[int, int], sample_size: int) -> np.ndarray:
        """
        Extract sample data from specified byte range for validation.
        
        Args:
            segy_handler: SEGY file handler
            byte_range: Tuple of (start_byte, end_byte)
            sample_size: Number of traces to sample
            
        Returns:
            Numpy array of extracted values
        """
        if not segy_handler.segy_file or not segy_handler.file_info:
            return np.array([])
        
        try:
            # Get representative sample of trace indices
            trace_indices = segy_handler.get_sample_trace_indices(sample_size)
            
            if len(trace_indices) == 0:
                return np.array([])
            
            # Extract trace header data from the specified byte range
            sample_data = segy_handler.get_trace_header_sample(trace_indices, byte_range)
            
            # Filter out obviously invalid values
            sample_data = self._filter_invalid_values(sample_data, byte_range)
            
            return sample_data
            
        except Exception as e:
            print(f"Error extracting sample data: {e}")
            return np.array([])
    
    def compute_statistics(self, data: np.ndarray, attribute_name: str) -> StatisticalProfile:
        """
        Compute comprehensive statistical profile of extracted data.
        
        Args:
            data: Numpy array of extracted values
            attribute_name: Name of the attribute being analyzed
            
        Returns:
            StatisticalProfile with comprehensive analysis
        """
        # Determine attribute type for specialized analysis
        attribute_type = self._determine_attribute_type(attribute_name)
        
        # Use statistical analyzer to compute profile
        profile = self.statistical_analyzer.analyze_data(data, attribute_type)
        
        return profile
    
    def validate_multiple_hypotheses(self, hypotheses: List[AttributeHypothesis], segy_handler: SEGYFileHandler) -> List[ValidationResult]:
        """
        Validate multiple hypotheses efficiently.
        
        Args:
            hypotheses: List of hypotheses to validate
            segy_handler: SEGY file handler
            
        Returns:
            List of validation results in the same order as input hypotheses
        """
        results = []
        
        for hypothesis in hypotheses:
            result = self.validate_hypothesis(hypothesis, segy_handler)
            results.append(result)
            
            # Update hypothesis validation status
            hypothesis.validation_status = ValidationStatus.PASSED if result.passed else ValidationStatus.FAILED
        
        return results
    
    def validate_with_cross_checks(self, hypotheses: List[AttributeHypothesis], segy_handler: SEGYFileHandler) -> List[ValidationResult]:
        """
        Validate hypotheses with cross-attribute consistency checks.
        
        Args:
            hypotheses: List of hypotheses to validate
            segy_handler: SEGY file handler
            
        Returns:
            List of validation results with cross-validation
        """
        # First, validate each hypothesis individually
        results = self.validate_multiple_hypotheses(hypotheses, segy_handler)
        
        # Extract statistical profiles for cross-validation
        attribute_stats = {}
        for i, hypothesis in enumerate(hypotheses):
            if results[i].passed:
                attribute_stats[hypothesis.attribute_name] = results[i].statistical_profile
        
        # Perform cross-attribute consistency validation
        if len(attribute_stats) > 1:
            consistency_results = self.validation_llm.validate_attribute_consistency(
                attribute_stats,
                context=f"SEGY file: {segy_handler.filepath}"
            )
            
            # Update results with consistency findings
            for i, hypothesis in enumerate(hypotheses):
                if hypothesis.attribute_name in consistency_results:
                    consistency_decision = consistency_results[hypothesis.attribute_name]
                    
                    # Update the validation result
                    results[i].passed = results[i].passed and consistency_decision.is_valid
                    results[i].confidence = min(results[i].confidence, consistency_decision.confidence)
                    results[i].issues.extend(consistency_decision.issues)
                    
                    # Update hypothesis status
                    hypothesis.validation_status = ValidationStatus.PASSED if results[i].passed else ValidationStatus.FAILED
        
        return results
    
    def create_validation_report(self, hypotheses: List[AttributeHypothesis], results: List[ValidationResult]) -> str:
        """
        Create a comprehensive validation report.
        
        Args:
            hypotheses: List of validated hypotheses
            results: List of validation results
            
        Returns:
            Formatted validation report string
        """
        report = "SEGY ATTRIBUTE VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Summary statistics
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        
        report += f"SUMMARY:\n"
        report += f"Total Attributes Validated: {total_count}\n"
        report += f"Passed Validation: {passed_count}\n"
        report += f"Failed Validation: {total_count - passed_count}\n"
        report += f"Success Rate: {passed_count/total_count*100:.1f}%\n\n"
        
        # Detailed results for each attribute
        report += "DETAILED RESULTS:\n"
        report += "-" * 30 + "\n\n"
        
        for i, (hypothesis, result) in enumerate(zip(hypotheses, results)):
            report += f"Attribute {i+1}: {hypothesis.attribute_name}\n"
            report += f"Byte Range: {hypothesis.byte_start}-{hypothesis.byte_end}\n"
            report += f"Data Type: {hypothesis.data_type}\n"
            report += f"Source: {hypothesis.source}\n"
            report += f"Status: {'PASSED' if result.passed else 'FAILED'}\n"
            report += f"Confidence: {result.confidence:.3f}\n"
            
            # Statistical summary
            stats = result.statistical_profile
            report += f"Statistics:\n"
            report += f"  Mean: {stats.mean:.6f}\n"
            report += f"  Range: {stats.max_val - stats.min_val:.6f}\n"
            report += f"  Std Dev: {stats.std:.6f}\n"
            report += f"  Data Type: {stats.detected_type}\n"
            report += f"  Outliers: {stats.outlier_count}\n"
            report += f"  Invalid Values: {stats.null_count}\n"
            
            # Issues and suggestions
            if result.issues:
                report += f"Issues:\n"
                for issue in result.issues:
                    report += f"  - {issue}\n"
            
            if result.suggestions:
                report += f"Suggestions:\n"
                for suggestion in result.suggestions:
                    report += f"  - {suggestion.suggestion_type}: {suggestion.reasoning}\n"
            
            report += f"LLM Evaluation: {result.llm_evaluation[:200]}...\n"
            report += "\n" + "-" * 30 + "\n\n"
        
        return report
    
    def _create_failed_validation(self, error_message: str) -> ValidationResult:
        """Create a failed validation result with error message"""
        empty_profile = StatisticalProfile(
            mean=0.0, median=0.0, std=0.0, min_val=0.0, max_val=0.0,
            skewness=0.0, kurtosis=0.0, percentiles={},
            is_monotonic=False, has_periodicity=False,
            outlier_count=0, null_count=0,
            detected_type='unknown', precision=0, has_invalid_values=True
        )
        
        return ValidationResult(
            passed=False,
            confidence=0.0,
            statistical_profile=empty_profile,
            llm_evaluation=error_message,
            issues=[error_message],
            suggestions=[]
        )
    
    def _filter_invalid_values(self, data: np.ndarray, byte_range: Tuple[int, int]) -> np.ndarray:
        """Filter out obviously invalid values from extracted data"""
        if len(data) == 0:
            return data
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(data)
        
        # Remove values that are likely parsing errors
        # (e.g., extremely large values that suggest wrong byte interpretation)
        abs_data = np.abs(data)
        reasonable_mask = abs_data < 1e15  # Very large threshold to catch obvious errors
        
        # Combine masks
        combined_mask = valid_mask & reasonable_mask
        
        return data[combined_mask]
    
    def _determine_attribute_type(self, attribute_name: str) -> str:
        """Determine the general type of attribute for specialized analysis"""
        attr_lower = attribute_name.lower()
        
        if any(coord in attr_lower for coord in ['x', 'y', 'z', 'coordinate', 'utm', 'geographic']):
            return 'coordinate'
        elif any(time_attr in attr_lower for time_attr in ['time', 'delay', 'sample', 'interval']):
            return 'time'
        elif any(amp in attr_lower for amp in ['amplitude', 'trace_data', 'scaling', 'gain']):
            return 'amplitude'
        elif any(idx in attr_lower for idx in ['inline', 'crossline', 'cdp', 'trace', 'shot', 'receiver']):
            return 'index'
        else:
            return 'general'
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Get summary statistics for a set of validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        
        confidences = [r.confidence for r in results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Count confidence levels
        high_conf = sum(1 for c in confidences if c >= self.high_confidence_threshold)
        medium_conf = sum(1 for c in confidences if self.medium_confidence_threshold <= c < self.high_confidence_threshold)
        low_conf = sum(1 for c in confidences if c < self.medium_confidence_threshold)
        
        # Common issues
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'total_validated': total_count,
            'passed': passed_count,
            'failed': total_count - passed_count,
            'success_rate': passed_count / total_count if total_count > 0 else 0.0,
            'average_confidence': avg_confidence,
            'confidence_distribution': {
                'high': high_conf,
                'medium': medium_conf,
                'low': low_conf
            },
            'common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def validate_with_iterative_refinement(self, hypothesis: AttributeHypothesis, segy_handler: SEGYFileHandler, max_iterations: int = 3) -> Tuple[ValidationResult, List[AttributeHypothesis]]:
        """
        Validate hypothesis with iterative refinement if initial validation fails.
        
        Args:
            hypothesis: Initial hypothesis to validate
            segy_handler: SEGY file handler
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            Tuple of (final_validation_result, list_of_attempted_hypotheses)
        """
        attempted_hypotheses = [hypothesis]
        current_hypothesis = hypothesis
        
        for iteration in range(max_iterations):
            # Validate current hypothesis
            result = self.validate_hypothesis(current_hypothesis, segy_handler)
            
            if result.passed:
                # Validation succeeded
                return result, attempted_hypotheses
            
            # If validation failed and we have more iterations, try refinements
            if iteration < max_iterations - 1 and result.suggestions:
                # Generate refined hypothesis based on suggestions
                refined_hypothesis = self._apply_refinement_suggestions(
                    current_hypothesis, 
                    result.suggestions
                )
                
                if refined_hypothesis:
                    attempted_hypotheses.append(refined_hypothesis)
                    current_hypothesis = refined_hypothesis
                else:
                    # No valid refinement could be generated
                    break
            else:
                # No more iterations or no suggestions
                break
        
        # Return the last validation result
        final_result = self.validate_hypothesis(current_hypothesis, segy_handler)
        return final_result, attempted_hypotheses
    
    def _apply_refinement_suggestions(self, original_hypothesis: AttributeHypothesis, suggestions: List) -> Optional[AttributeHypothesis]:
        """Apply refinement suggestions to create a new hypothesis"""
        if not suggestions:
            return None
        
        # Take the highest confidence suggestion
        best_suggestion = max(suggestions, key=lambda s: s.confidence)
        
        # Create new hypothesis based on suggestion type
        new_hypothesis = AttributeHypothesis(
            attribute_name=original_hypothesis.attribute_name,
            byte_start=original_hypothesis.byte_start,
            byte_end=original_hypothesis.byte_end,
            confidence=original_hypothesis.confidence * 0.8,  # Lower confidence for refined hypothesis
            data_type=original_hypothesis.data_type,
            source=f"{original_hypothesis.source}_refined",
            reasoning=f"Refined based on: {best_suggestion.reasoning}",
            validation_status=ValidationStatus.NOT_VALIDATED
        )
        
        # Apply specific refinements based on suggestion type
        if best_suggestion.suggestion_type == 'byte_range':
            # Adjust byte range (simplified implementation)
            range_size = original_hypothesis.byte_end - original_hypothesis.byte_start
            new_hypothesis.byte_start = max(0, original_hypothesis.byte_start - 2)
            new_hypothesis.byte_end = new_hypothesis.byte_start + range_size
            
        elif best_suggestion.suggestion_type == 'data_type':
            # Try alternative data type
            current_type = original_hypothesis.data_type
            if current_type == 'int32':
                new_hypothesis.data_type = 'float32'
            elif current_type == 'float32':
                new_hypothesis.data_type = 'int16'
            elif current_type == 'int16':
                new_hypothesis.data_type = 'int32'
            
        elif best_suggestion.suggestion_type == 'location':
            # Explore nearby location
            new_hypothesis.byte_start = original_hypothesis.byte_start + 4
            new_hypothesis.byte_end = original_hypothesis.byte_end + 4
            
        elif best_suggestion.suggestion_type == 'alignment':
            # Fix byte alignment
            aligned_start = (original_hypothesis.byte_start // 4) * 4
            range_size = original_hypothesis.byte_end - original_hypothesis.byte_start
            new_hypothesis.byte_start = aligned_start
            new_hypothesis.byte_end = aligned_start + range_size
        
        return new_hypothesis