"""
Core data models for the AI-Powered SEGY Metadata Parser system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np


class ValidationStatus(Enum):
    """Status of hypothesis validation"""
    NOT_VALIDATED = "not_validated"
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"


class ConfidenceLevel(Enum):
    """Confidence levels for hypotheses and results"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValidationRule:
    """Defines validation criteria for attributes"""
    rule_type: str  # 'range', 'pattern', 'data_type'
    parameters: Dict[str, Any]
    description: str


@dataclass
class AttributeSchema:
    """Schema definition for SEGY attributes"""
    name: str
    expected_types: List[str]  # ['int32', 'float32']
    expected_range: Tuple[float, float]
    standard_locations: Dict[str, List[Tuple[int, int]]]  # revision -> byte ranges
    validation_rules: List[ValidationRule]
    cross_checks: List[str] = field(default_factory=list)  # related attributes
    description: str = ""


@dataclass
class AttributeHypothesis:
    """Represents a hypothesis about an attribute's byte location"""
    attribute_name: str
    byte_start: int
    byte_end: int
    confidence: float  # 0.0 to 1.0
    data_type: str  # 'int16', 'int32', 'float32', etc.
    source: str  # 'textual_header', 'binary_header', 'fallback', etc.
    reasoning: str  # LLM reasoning for this hypothesis
    validation_status: Optional[ValidationStatus] = ValidationStatus.NOT_VALIDATED


@dataclass
class StatisticalProfile:
    """Comprehensive statistical analysis of extracted data"""
    # Basic statistics
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    percentiles: Dict[int, float]  # 5th, 25th, 75th, 95th
    
    # Pattern analysis
    is_monotonic: bool
    has_periodicity: bool
    outlier_count: int
    null_count: int
    
    # Data type analysis
    detected_type: str
    precision: int
    has_invalid_values: bool


@dataclass
class RefinementSuggestion:
    """Suggestion for refining a failed hypothesis"""
    suggestion_type: str  # 'byte_range', 'data_type', 'location'
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float


@dataclass
class ValidationResult:
    """Result of validating a hypothesis against actual data"""
    passed: bool
    confidence: float
    statistical_profile: StatisticalProfile
    llm_evaluation: str
    issues: List[str]
    suggestions: List[RefinementSuggestion]


@dataclass
class ReasoningResult:
    """Result of chain-of-thought reasoning process"""
    conclusion: str
    reasoning_steps: List[str]
    confidence: float
    alternative_interpretations: List[str]
    requires_validation: bool


@dataclass
class RevisionInfo:
    """Information about detected SEGY revision"""
    revision: str  # '0', '1.0', '2.0', '2.1', 'None'
    confidence: ConfidenceLevel
    source: str  # 'textual_header', 'binary_header', 'inferred'
    reasoning: str = ""


@dataclass
class GeometricInfo:
    """Geometric information extracted from SEGY file"""
    world_coordinates: Dict[str, AttributeHypothesis]  # X, Y, Z mappings
    azimuthal_angle: Optional[AttributeHypothesis]
    inline_crossline: Dict[str, AttributeHypothesis]  # inline, crossline mappings
    coordinate_reference_system: Optional[str]
    projection_info: Optional[str]


@dataclass
class ParsedResults:
    """Complete results of SEGY file parsing"""
    filename: str
    revision_info: RevisionInfo
    attributes: List[AttributeHypothesis]
    geometric_info: GeometricInfo
    validation_results: List[ValidationResult]
    reasoning_chains: List[ReasoningResult]
    fallback_strategies_used: List[str]
    processing_time: float
    confidence_summary: Dict[str, int]  # high, medium, low counts


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    model: str
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SEGYFileInfo:
    """Basic information about a SEGY file"""
    filepath: str
    file_size: int
    trace_count: int
    sample_count: int
    sample_interval: float
    format_code: int
    has_textual_headers: bool
    has_binary_header: bool