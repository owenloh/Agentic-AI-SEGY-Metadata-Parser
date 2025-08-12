"""
ValidationLLM - LLM-based evaluation of whether statistical patterns match expected attribute behavior.

This module uses LLM reasoning to evaluate if extracted data patterns are consistent
with the expected behavior of specific SEGY attributes.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from models.data_models import StatisticalProfile, RefinementSuggestion
from core.llm_provider import LLMFactory, LLMProvider


@dataclass
class ValidationDecision:
    """Decision from LLM validation of statistical patterns"""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    issues: List[str]
    suggestions: List[str]
    attribute_likelihood: float  # How likely this data represents the claimed attribute


class ValidationLLM:
    """LLM-based evaluation of whether statistical patterns match expected attribute behavior"""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.max_tokens = 4000
        self.temperature = 0.3  # Lower temperature for more consistent validation
        
        # Define expected patterns for different attribute types
        self.attribute_expectations = {
            'coordinate': {
                'x': {'range': (0, 10000000), 'precision': 2, 'type': 'float'},
                'y': {'range': (0, 10000000), 'precision': 2, 'type': 'float'},
                'z': {'range': (-5000, 5000), 'precision': 2, 'type': 'float'}
            },
            'time': {
                'sample_interval': {'range': (0.1, 10), 'precision': 3, 'type': 'float'},
                'delay': {'range': (0, 10000), 'precision': 1, 'type': 'float'},
                'recording_time': {'range': (0, 86400), 'precision': 0, 'type': 'int'}
            },
            'amplitude': {
                'trace_data': {'range': (-1e6, 1e6), 'precision': 6, 'type': 'float'},
                'scaling_factor': {'range': (1, 10000), 'precision': 0, 'type': 'int'}
            },
            'index': {
                'inline': {'range': (1, 100000), 'precision': 0, 'type': 'int'},
                'crossline': {'range': (1, 100000), 'precision': 0, 'type': 'int'},
                'cdp': {'range': (1, 1000000), 'precision': 0, 'type': 'int'},
                'trace_number': {'range': (1, 1000000), 'precision': 0, 'type': 'int'}
            }
        }
    
    def evaluate_statistics(self, stats: StatisticalProfile, attribute_name: str, context: str = "") -> ValidationDecision:
        """
        Evaluate if statistical patterns match expected attribute behavior.
        
        Args:
            stats: Statistical profile of extracted data
            attribute_name: Name of the attribute being validated
            context: Additional context about the SEGY file or parsing situation
            
        Returns:
            ValidationDecision with evaluation results
        """
        provider = self.llm_factory.get_available_provider()
        if not provider:
            return ValidationDecision(
                is_valid=False,
                confidence=0.0,
                reasoning="No LLM provider available for validation",
                issues=["LLM provider unavailable"],
                suggestions=["Check LLM configuration"],
                attribute_likelihood=0.0
            )
        
        # Create validation prompt
        validation_prompt = self._create_validation_prompt(stats, attribute_name, context)
        
        # Get LLM evaluation
        response = provider.invoke_prompt(
            system_prompt=self._get_validation_system_prompt(),
            user_prompt=validation_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if not response.success:
            return ValidationDecision(
                is_valid=False,
                confidence=0.0,
                reasoning=f"LLM validation failed: {response.error_message}",
                issues=["LLM validation error"],
                suggestions=["Retry validation or use fallback methods"],
                attribute_likelihood=0.0
            )
        
        # Parse LLM response
        return self._parse_validation_response(response.content, stats, attribute_name)
    
    def explain_decision(self, decision: ValidationDecision) -> str:
        """
        Provide detailed explanation of validation decision.
        
        Args:
            decision: ValidationDecision to explain
            
        Returns:
            Detailed explanation string
        """
        explanation = f"Validation Decision: {'VALID' if decision.is_valid else 'INVALID'}\n"
        explanation += f"Confidence: {decision.confidence:.2f}\n\n"
        explanation += f"Reasoning:\n{decision.reasoning}\n\n"
        
        if decision.issues:
            explanation += "Issues Identified:\n"
            for issue in decision.issues:
                explanation += f"- {issue}\n"
            explanation += "\n"
        
        if decision.suggestions:
            explanation += "Suggestions:\n"
            for suggestion in decision.suggestions:
                explanation += f"- {suggestion}\n"
            explanation += "\n"
        
        explanation += f"Attribute Likelihood: {decision.attribute_likelihood:.2f}"
        
        return explanation
    
    def suggest_refinements(self, failed_validation: ValidationDecision, stats: StatisticalProfile, attribute_name: str) -> List[RefinementSuggestion]:
        """
        Generate refinement suggestions based on failed validation.
        
        Args:
            failed_validation: The failed validation decision
            stats: Statistical profile that failed validation
            attribute_name: Name of the attribute that failed
            
        Returns:
            List of refinement suggestions
        """
        provider = self.llm_factory.get_available_provider()
        if not provider:
            return []
        
        refinement_prompt = self._create_refinement_prompt(failed_validation, stats, attribute_name)
        
        response = provider.invoke_prompt(
            system_prompt=self._get_refinement_system_prompt(),
            user_prompt=refinement_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if not response.success:
            return []
        
        return self._parse_refinement_response(response.content)
    
    def _create_validation_prompt(self, stats: StatisticalProfile, attribute_name: str, context: str) -> str:
        """Create prompt for statistical validation"""
        
        # Get expected patterns for this attribute type
        expected_info = self._get_expected_patterns(attribute_name)
        
        prompt = f"""
        I need you to evaluate whether statistical patterns from extracted SEGY data are consistent with the expected behavior of a specific attribute.

        ATTRIBUTE: {attribute_name}
        CONTEXT: {context}

        EXTRACTED DATA STATISTICS:
        - Mean: {stats.mean:.6f}
        - Median: {stats.median:.6f}
        - Standard Deviation: {stats.std:.6f}
        - Min: {stats.min_val:.6f}
        - Max: {stats.max_val:.6f}
        - Range: {stats.max_val - stats.min_val:.6f}
        - Skewness: {stats.skewness:.3f}
        - Kurtosis: {stats.kurtosis:.3f}
        - Detected Type: {stats.detected_type}
        - Precision: {stats.precision}
        - Is Monotonic: {stats.is_monotonic}
        - Has Periodicity: {stats.has_periodicity}
        - Outlier Count: {stats.outlier_count}
        - Null Count: {stats.null_count}
        - Has Invalid Values: {stats.has_invalid_values}

        PERCENTILES:
        """
        
        for percentile, value in stats.percentiles.items():
            prompt += f"        - {percentile}th: {value:.6f}\n"
        
        prompt += f"""

        EXPECTED PATTERNS FOR {attribute_name.upper()}:
        {expected_info}

        Please evaluate:
        1. Does this data pattern match what you'd expect for a '{attribute_name}' attribute in SEGY files?
        2. Are the statistical properties reasonable for this attribute type?
        3. What is your confidence level (0.0-1.0) in this validation?
        4. What specific issues do you identify, if any?
        5. What is the likelihood (0.0-1.0) that this data actually represents the claimed attribute?

        Provide your analysis in this format:
        VALID: [true/false]
        CONFIDENCE: [0.0-1.0]
        ATTRIBUTE_LIKELIHOOD: [0.0-1.0]
        REASONING: [detailed explanation]
        ISSUES: [list of specific issues, or "none"]
        SUGGESTIONS: [list of suggestions for improvement, or "none"]
        """
        
        return prompt
    
    def _create_refinement_prompt(self, failed_validation: ValidationDecision, stats: StatisticalProfile, attribute_name: str) -> str:
        """Create prompt for generating refinement suggestions"""
        
        prompt = f"""
        A SEGY attribute validation has failed and I need suggestions for refinement.

        ATTRIBUTE: {attribute_name}
        VALIDATION FAILURE REASON: {failed_validation.reasoning}
        CONFIDENCE: {failed_validation.confidence:.2f}
        ATTRIBUTE LIKELIHOOD: {failed_validation.attribute_likelihood:.2f}

        ISSUES IDENTIFIED:
        """
        
        for issue in failed_validation.issues:
            prompt += f"- {issue}\n"
        
        prompt += f"""

        STATISTICAL PROFILE:
        - Mean: {stats.mean:.6f}
        - Range: {stats.max_val - stats.min_val:.6f}
        - Data Type: {stats.detected_type}
        - Has Invalid Values: {stats.has_invalid_values}
        - Outlier Count: {stats.outlier_count}

        Please provide specific refinement suggestions to address the validation failure:

        1. BYTE_RANGE adjustments (if the issue might be incorrect byte boundaries)
        2. DATA_TYPE alternatives (if the data type interpretation seems wrong)
        3. LOCATION exploration (if we should try nearby byte locations)
        4. ALIGNMENT fixes (if byte alignment might be the issue)
        5. ENDIANNESS considerations (if byte order might be wrong)

        For each suggestion, provide:
        - Type of refinement
        - Specific parameters or changes to try
        - Reasoning for why this might fix the issue
        - Confidence in the suggestion (0.0-1.0)

        Format your response as:
        SUGGESTION_1:
        TYPE: [BYTE_RANGE/DATA_TYPE/LOCATION/ALIGNMENT/ENDIANNESS]
        PARAMETERS: [specific changes to make]
        REASONING: [why this might work]
        CONFIDENCE: [0.0-1.0]

        SUGGESTION_2:
        [continue pattern...]
        """
        
        return prompt
    
    def _get_validation_system_prompt(self) -> str:
        """Get system prompt for validation"""
        return """You are an expert in SEGY file format analysis and statistical validation. Your role is to evaluate whether extracted data patterns are consistent with expected SEGY attribute behavior.

        You have deep knowledge of:
        - SEGY file format standards and conventions
        - Expected data ranges and patterns for different attribute types
        - Statistical analysis and data quality assessment
        - Common issues in SEGY file parsing and data extraction

        When evaluating data patterns:
        - Consider the specific attribute type and its expected characteristics
        - Look for statistical anomalies that might indicate parsing errors
        - Assess data quality and consistency
        - Provide clear, actionable feedback
        - Be conservative in validation - it's better to flag potential issues than miss them

        Always provide specific, technical reasoning for your decisions."""
    
    def _get_refinement_system_prompt(self) -> str:
        """Get system prompt for refinement suggestions"""
        return """You are an expert in SEGY file parsing and data extraction troubleshooting. Your role is to provide specific, actionable suggestions for fixing validation failures.

        You understand:
        - Common causes of SEGY parsing failures
        - Byte alignment and endianness issues
        - Data type interpretation problems
        - Standard and non-standard byte locations for SEGY attributes
        - Refinement strategies that are most likely to succeed

        When providing refinement suggestions:
        - Be specific about what changes to make
        - Prioritize the most likely solutions first
        - Consider multiple potential causes
        - Provide clear reasoning for each suggestion
        - Include confidence estimates for your suggestions

        Focus on practical, implementable solutions."""
    
    def _get_expected_patterns(self, attribute_name: str) -> str:
        """Get expected patterns description for an attribute"""
        
        # Normalize attribute name
        attr_lower = attribute_name.lower()
        
        # Coordinate attributes
        if any(coord in attr_lower for coord in ['x', 'source_x', 'group_x', 'cdp_x']):
            return """
            X-coordinates in SEGY files typically:
            - Range from 0 to 10,000,000 (UTM coordinates) or -180 to 180 (geographic)
            - Are float32 or int32 values
            - Have 2-6 decimal places for UTM, more for geographic
            - Show spatial clustering or regular patterns
            - Should not have excessive outliers or invalid values
            """
        
        elif any(coord in attr_lower for coord in ['y', 'source_y', 'group_y', 'cdp_y']):
            return """
            Y-coordinates in SEGY files typically:
            - Range from 0 to 10,000,000 (UTM coordinates) or -90 to 90 (geographic)
            - Are float32 or int32 values
            - Have 2-6 decimal places for UTM, more for geographic
            - Show spatial clustering or regular patterns
            - Should not have excessive outliers or invalid values
            """
        
        elif 'z' in attr_lower or 'elevation' in attr_lower:
            return """
            Z-coordinates/elevations in SEGY files typically:
            - Range from -5000 to 5000 meters
            - Are float32 values with 1-3 decimal places
            - May show topographic patterns
            - Should be relatively stable within a survey area
            """
        
        # Index attributes
        elif any(idx in attr_lower for idx in ['inline', 'crossline', 'cdp', 'trace']):
            return """
            Index values (inline, crossline, CDP, trace numbers) typically:
            - Are positive integers starting from 1
            - Show sequential or regular stepping patterns
            - Range from 1 to 100,000+ depending on survey size
            - Should be monotonic or have regular increments
            - Should not have gaps or invalid values
            """
        
        # Time attributes
        elif any(time_attr in attr_lower for time_attr in ['time', 'delay', 'sample']):
            return """
            Time-related attributes typically:
            - Sample intervals: 0.1 to 10 milliseconds
            - Delays: 0 to 10,000 milliseconds
            - Recording times: 0 to 86,400 seconds (24 hours)
            - Are usually float32 or int32 values
            - Should have consistent precision and reasonable ranges
            """
        
        # Amplitude/trace data
        elif any(amp in attr_lower for amp in ['amplitude', 'trace_data', 'scaling']):
            return """
            Amplitude/trace data typically:
            - Wide dynamic range (-1e6 to 1e6)
            - Float32 values with variable precision
            - May have zero crossings and oscillatory patterns
            - Scaling factors are usually small positive integers
            - Should not be all zeros or constant values
            """
        
        else:
            return f"""
            For attribute '{attribute_name}':
            - Should have reasonable data ranges for the attribute type
            - Should not have excessive invalid or null values
            - Data type should be appropriate (int16/32, float32/64)
            - Statistical distribution should make sense for the attribute
            - Should not have obvious parsing artifacts or errors
            """
    
    def _parse_validation_response(self, response_content: str, stats: StatisticalProfile, attribute_name: str) -> ValidationDecision:
        """Parse LLM validation response into ValidationDecision"""
        
        # Initialize default values
        is_valid = False
        confidence = 0.0
        attribute_likelihood = 0.0
        reasoning = response_content
        issues = []
        suggestions = []
        
        try:
            lines = response_content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('VALID:'):
                    is_valid = 'true' in line.lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except:
                        confidence = 0.5
                elif line.startswith('ATTRIBUTE_LIKELIHOOD:'):
                    try:
                        attribute_likelihood = float(line.split(':', 1)[1].strip())
                    except:
                        attribute_likelihood = 0.5
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
                elif line.startswith('ISSUES:'):
                    issues_text = line.split(':', 1)[1].strip()
                    if issues_text.lower() != 'none':
                        issues = [issue.strip() for issue in issues_text.split(',')]
                elif line.startswith('SUGGESTIONS:'):
                    suggestions_text = line.split(':', 1)[1].strip()
                    if suggestions_text.lower() != 'none':
                        suggestions = [suggestion.strip() for suggestion in suggestions_text.split(',')]
        
        except Exception as e:
            # If parsing fails, use conservative defaults
            reasoning = f"Failed to parse LLM response: {str(e)}\nOriginal response: {response_content}"
            issues = ["Response parsing error"]
        
        return ValidationDecision(
            is_valid=is_valid,
            confidence=confidence,
            reasoning=reasoning,
            issues=issues,
            suggestions=suggestions,
            attribute_likelihood=attribute_likelihood
        )
    
    def _parse_refinement_response(self, response_content: str) -> List[RefinementSuggestion]:
        """Parse LLM refinement response into RefinementSuggestion list"""
        
        suggestions = []
        
        try:
            # Split response into suggestion blocks
            blocks = response_content.split('SUGGESTION_')
            
            for block in blocks[1:]:  # Skip first empty block
                lines = block.strip().split('\n')
                
                suggestion_type = ""
                parameters = {}
                reasoning = ""
                confidence = 0.5
                
                for line in lines:
                    line = line.strip()
                    
                    if line.startswith('TYPE:'):
                        suggestion_type = line.split(':', 1)[1].strip()
                    elif line.startswith('PARAMETERS:'):
                        params_text = line.split(':', 1)[1].strip()
                        # Simple parameter parsing - could be enhanced
                        parameters = {'description': params_text}
                    elif line.startswith('REASONING:'):
                        reasoning = line.split(':', 1)[1].strip()
                    elif line.startswith('CONFIDENCE:'):
                        try:
                            confidence = float(line.split(':', 1)[1].strip())
                        except:
                            confidence = 0.5
                
                if suggestion_type and reasoning:
                    suggestions.append(RefinementSuggestion(
                        suggestion_type=suggestion_type.lower(),
                        parameters=parameters,
                        reasoning=reasoning,
                        confidence=confidence
                    ))
        
        except Exception as e:
            # If parsing fails, return empty list
            print(f"Failed to parse refinement suggestions: {e}")
        
        return suggestions
    
    def validate_attribute_consistency(self, attribute_stats: Dict[str, StatisticalProfile], context: str = "") -> Dict[str, ValidationDecision]:
        """
        Validate consistency across multiple related attributes.
        
        Args:
            attribute_stats: Dictionary mapping attribute names to their statistical profiles
            context: Additional context about the validation
            
        Returns:
            Dictionary mapping attribute names to validation decisions
        """
        results = {}
        
        # First, validate each attribute individually
        for attr_name, stats in attribute_stats.items():
            results[attr_name] = self.evaluate_statistics(stats, attr_name, context)
        
        # Then check for cross-attribute consistency
        consistency_issues = self._check_cross_attribute_consistency(attribute_stats)
        
        # Update results with consistency issues
        for attr_name, issues in consistency_issues.items():
            if attr_name in results:
                results[attr_name].issues.extend(issues)
                # Lower confidence if there are consistency issues
                if issues:
                    results[attr_name].confidence *= 0.8
                    results[attr_name].is_valid = results[attr_name].is_valid and len(issues) == 0
        
        return results
    
    def _check_cross_attribute_consistency(self, attribute_stats: Dict[str, StatisticalProfile]) -> Dict[str, List[str]]:
        """Check consistency between related attributes"""
        issues = {attr: [] for attr in attribute_stats.keys()}
        
        # Check coordinate consistency
        coord_attrs = {k: v for k, v in attribute_stats.items() if any(coord in k.lower() for coord in ['x', 'y', 'z'])}
        if len(coord_attrs) > 1:
            coord_ranges = {k: v.max_val - v.min_val for k, v in coord_attrs.items()}
            
            # Check if coordinate ranges are reasonable relative to each other
            ranges = list(coord_ranges.values())
            if max(ranges) / min(ranges) > 1000:  # Very different scales
                for attr in coord_attrs.keys():
                    issues[attr].append("Coordinate ranges inconsistent with other coordinates")
        
        # Check index consistency
        index_attrs = {k: v for k, v in attribute_stats.items() if any(idx in k.lower() for idx in ['inline', 'crossline', 'trace'])}
        if len(index_attrs) > 1:
            for attr_name, stats in index_attrs.items():
                if not stats.is_monotonic and 'trace' in attr_name.lower():
                    issues[attr_name].append("Trace numbers should typically be monotonic")
        
        return issues