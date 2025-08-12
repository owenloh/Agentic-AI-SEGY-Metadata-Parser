"""
Enhanced SEGY Header Parser - Main orchestrator integrating all components.

This module provides the main SEGYHeaderParser class that orchestrates the complete
parsing workflow from textual header analysis through validation to result export.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from core.llm_header_parser import LLMHeaderParser
from core.attribute_ontology import AttributeOntology
from core.segy_file_handler import SEGYFileHandler
from core.trace_data_validator import TraceDataValidator
from core.statistical_analyzer import StatisticalAnalyzer
from core.validation_llm import ValidationLLM
from core.geometric_extractor import GeometricExtractor
from core.result_exporter import ResultExporter
from core.fallback_strategy_manager import FallbackStrategyManager
from core.hypothesis_refiner import HypothesisRefiner
from core.llm_provider import LLMFactory, ChainOfThoughtReasoner
from models.data_models import (
    ParsedResults, AttributeHypothesis, ValidationResult, 
    GeometricInfo, ReasoningResult, RevisionInfo, ConfidenceLevel
)


@dataclass
class ParsingConfig:
    """Configuration for the parsing process"""
    max_validation_attempts: int = 3
    sample_size_for_validation: int = 100
    enable_chain_of_thought: bool = True
    enable_fallback_strategies: bool = True
    enable_hypothesis_refinement: bool = True
    skip_validation_for_explicit: bool = True  # Skip validation for explicit high-confidence mappings
    explicit_confidence_threshold: float = 0.85  # Confidence threshold for skipping validation
    output_formats: List[str] = None  # ['json', 'txt', 'csv']
    verbose_logging: bool = False
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['json', 'txt', 'csv']


class SEGYHeaderParser:
    """
    Enhanced SEGY Header Parser - Main orchestrator class.
    
    Integrates all components to provide a complete parsing workflow:
    1. Textual header parsing with LLM
    2. Hypothesis generation and ontology guidance
    3. Data validation against actual trace data
    4. Geometric information extraction
    5. Chain-of-thought reasoning for complex cases
    6. Fallback strategies and hypothesis refinement
    7. Comprehensive result export
    """
    
    def __init__(self, config: Optional[ParsingConfig] = None):
        self.config = config or ParsingConfig()
        
        # Configure logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.logger.info("Initializing AI-Powered SEGY Metadata Parser components...")
        
        try:
            # Core system components
            self.ontology = AttributeOntology()
            self.llm_factory = LLMFactory()
            
            # Parsing and validation components
            self.header_parser = LLMHeaderParser(self.llm_factory, self.ontology)
            self.trace_validator = TraceDataValidator(self.llm_factory)
            self.geometric_extractor = GeometricExtractor(self.ontology)
            
            # Advanced reasoning components
            self.chain_reasoner = ChainOfThoughtReasoner(self.llm_factory)
            self.fallback_manager = FallbackStrategyManager(self.ontology)
            self.hypothesis_refiner = HypothesisRefiner(self.llm_factory, self.ontology)
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def parse_segy_file(self, segy_file_path: Path, output_dir: Optional[Path] = None) -> ParsedResults:
        """
        Parse a single SEGY file through the complete workflow.
        
        Args:
            segy_file_path: Path to the SEGY file to parse
            output_dir: Optional output directory for results (defaults to current dir)
            
        Returns:
            ParsedResults with complete parsing information
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting enhanced parsing of: {segy_file_path.name}")
        
        # Initialize result tracking
        reasoning_chains = []
        fallback_strategies_used = []
        validation_results = []
        
        try:
            # Step 1: Parse textual headers with LLM
            self.logger.info("📋 Step 1: Parsing textual headers...")
            header_result = self.header_parser.parse_header(segy_file_path)
            
            if not header_result.parsing_success:
                raise Exception(f"Header parsing failed: {header_result.error_message}")
            
            self.logger.info(f"✅ Header parsing completed")
            self.logger.info(f"   Revision: {header_result.revision_info.revision}")
            self.logger.info(f"   Initial hypotheses: {len(header_result.attribute_hypotheses)}")
            
            # Step 2: Apply fallback strategies if needed
            if self.config.enable_fallback_strategies and len(header_result.attribute_hypotheses) < 5:
                self.logger.info("🔄 Step 2: Applying fallback strategies...")
                fallback_hypotheses = self.fallback_manager.generate_fallback_hypotheses(
                    segy_file_path, header_result.revision_info, header_result.attribute_hypotheses
                )
                
                if fallback_hypotheses:
                    header_result.attribute_hypotheses.extend(fallback_hypotheses)
                    fallback_strategies_used.append("standard_byte_locations")
                    self.logger.info(f"✅ Added {len(fallback_hypotheses)} fallback hypotheses")
            
            # Step 3: Validate hypotheses against actual data
            self.logger.info("🔬 Step 3: Validating hypotheses with trace data...")
            segy_handler = SEGYFileHandler(segy_file_path)
            
            if not segy_handler.open_file():
                raise Exception("Failed to open SEGY file for validation")
            
            validated_hypotheses = []
            
            for i, hypothesis in enumerate(header_result.attribute_hypotheses):
                self.logger.info(f"   Validating {i+1}/{len(header_result.attribute_hypotheses)}: {hypothesis.attribute_name}")
                
                # Skip validation for high-confidence explicit attributes (if enabled)
                if (self.config.skip_validation_for_explicit and 
                    hypothesis.confidence >= self.config.explicit_confidence_threshold and 
                    self._is_explicit_mapping(hypothesis)):
                    self.logger.info(f"     ✅ Skipped validation (high confidence explicit mapping: {hypothesis.confidence:.3f})")
                    hypothesis.validation_status = "validated"
                    validated_hypotheses.append(hypothesis)
                    
                    # Create a mock validation result for consistency
                    from models.data_models import ValidationResult, StatisticalProfile
                    mock_validation = ValidationResult(
                        passed=True,
                        confidence=hypothesis.confidence,
                        statistical_profile=StatisticalProfile(
                            mean=0.0, median=0.0, std=0.0, min_val=0.0, max_val=0.0,
                            skewness=0.0, kurtosis=0.0, percentiles={}, is_monotonic=False,
                            has_periodicity=False, outlier_count=0, null_count=0,
                            detected_type=hypothesis.data_type, precision=0, has_invalid_values=False
                        ),
                        llm_evaluation="Skipped - explicit textual header mapping",
                        issues=[],
                        suggestions=[]
                    )
                    validation_results.append(mock_validation)
                    continue
                
                # Perform validation for other attributes
                validation_result = self._validate_with_refinement(
                    hypothesis, segy_handler, reasoning_chains
                )
                
                validation_results.append(validation_result)
                
                if validation_result.passed:
                    hypothesis.validation_status = "validated"
                    validated_hypotheses.append(hypothesis)
                    self.logger.info(f"     ✅ Validated (confidence: {validation_result.confidence:.3f})")
                else:
                    self.logger.info(f"     ❌ Failed validation")
            
            self.logger.info(f"✅ Validation completed: {len(validated_hypotheses)}/{len(header_result.attribute_hypotheses)} passed")
            
            # Step 4: Extract and enhance geometric information
            self.logger.info("🗺️  Step 4: Extracting geometric information...")
            geometric_info = self.geometric_extractor.extract_geometric_info(
                segy_handler, validated_hypotheses, header_result.revision_info, 
                header_geometric_data=header_result.geometric_data
            )
            
            self.logger.info(f"✅ Geometric extraction completed")
            self.logger.info(f"   World coordinates: {len(geometric_info.world_coordinates)}")
            self.logger.info(f"   Inline/crossline: {len(geometric_info.inline_crossline)}")
            
            # Log enhanced geometric information if available
            if header_result.geometric_data:
                geo_data = header_result.geometric_data
                self.logger.info(f"   Corner points: {len(geo_data.corner_points)}")
                self.logger.info(f"   Survey geometry: {'✅' if geo_data.survey_geometry else '❌'}")
                self.logger.info(f"   Coordinate system: {'✅' if geo_data.coordinate_system else '❌'}")
                self.logger.info(f"   Processing steps: {len(geo_data.processing_info)}")
            
            # Step 5: Generate comprehensive results
            processing_time = time.time() - start_time
            confidence_summary = self._calculate_confidence_summary(validated_hypotheses)
            
            parsed_results = ParsedResults(
                filename=segy_file_path.name,
                revision_info=header_result.revision_info,
                attributes=header_result.attribute_hypotheses,
                geometric_info=geometric_info,
                validation_results=validation_results,
                reasoning_chains=reasoning_chains,
                fallback_strategies_used=fallback_strategies_used,
                processing_time=processing_time,
                confidence_summary=confidence_summary
            )
            
            # Step 6: Export results
            if output_dir:
                self.logger.info("📤 Step 6: Exporting results...")
                exporter = ResultExporter(output_dir)
                exported_files = exporter.export_results(parsed_results)
                
                self.logger.info(f"✅ Results exported to {len(exported_files)} formats:")
                for fmt, path in exported_files.items():
                    self.logger.info(f"   {fmt.upper()}: {path}")
            
            segy_handler.close_file()
            
            self.logger.info(f"🎉 Parsing completed successfully in {processing_time:.2f}s")
            return parsed_results
            
        except Exception as e:
            self.logger.error(f"❌ Parsing failed: {e}")
            
            # Return partial results if possible
            processing_time = time.time() - start_time
            
            return ParsedResults(
                filename=segy_file_path.name,
                revision_info=RevisionInfo("None", ConfidenceLevel.LOW, "error", str(e)),
                attributes=[],
                geometric_info=GeometricInfo({}, None, {}, None, None),
                validation_results=[],
                reasoning_chains=reasoning_chains,
                fallback_strategies_used=fallback_strategies_used,
                processing_time=processing_time,
                confidence_summary={'high': 0, 'medium': 0, 'low': 0}
            )
    
    def parse_multiple_files(self, segy_files: List[Path], output_dir: Path) -> Dict[str, ParsedResults]:
        """
        Parse multiple SEGY files with progress tracking.
        
        Args:
            segy_files: List of SEGY file paths to parse
            output_dir: Output directory for all results
            
        Returns:
            Dictionary mapping filenames to ParsedResults
        """
        self.logger.info(f"🚀 Starting batch processing of {len(segy_files)} SEGY files")
        
        results = {}
        successful_parses = 0
        
        for i, segy_file in enumerate(segy_files, 1):
            self.logger.info(f"\n📁 Processing file {i}/{len(segy_files)}: {segy_file.name}")
            
            try:
                # Create subdirectory for this file's results
                file_output_dir = output_dir / segy_file.stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Parse the file
                result = self.parse_segy_file(segy_file, file_output_dir)
                results[segy_file.name] = result
                
                if len(result.attributes) > 0:
                    successful_parses += 1
                    self.logger.info(f"✅ File {i} completed successfully")
                else:
                    self.logger.warning(f"⚠️  File {i} completed with no attributes extracted")
                
            except Exception as e:
                self.logger.error(f"❌ File {i} failed: {e}")
                # Continue with next file
                continue
        
        self.logger.info(f"\n🎉 Batch processing completed: {successful_parses}/{len(segy_files)} files successful")
        return results
    
    def _validate_with_refinement(self, hypothesis: AttributeHypothesis, 
                                segy_handler: SEGYFileHandler, 
                                reasoning_chains: List[ReasoningResult]) -> ValidationResult:
        """Validate hypothesis with iterative refinement if needed."""
        
        for attempt in range(self.config.max_validation_attempts):
            # Validate current hypothesis
            validation_result = self.trace_validator.validate_hypothesis(hypothesis, segy_handler)
            
            if validation_result.passed:
                return validation_result
            
            # If validation failed and refinement is enabled, try to refine
            if self.config.enable_hypothesis_refinement and attempt < self.config.max_validation_attempts - 1:
                self.logger.info(f"     🔄 Refining hypothesis (attempt {attempt + 1})")
                
                # Use chain-of-thought reasoning for complex cases
                if self.config.enable_chain_of_thought:
                    reasoning_result = self.chain_reasoner.generate_alternatives(
                        hypothesis.__dict__
                    )
                    reasoning_chains.append(reasoning_result)
                
                # Refine the hypothesis
                refined_hypotheses = self.hypothesis_refiner.refine_hypothesis(
                    hypothesis, validation_result
                )
                
                if refined_hypotheses:
                    # Try the first refined hypothesis
                    hypothesis = refined_hypotheses[0]
                    self.logger.info(f"     🔧 Trying refined hypothesis: bytes {hypothesis.byte_start}-{hypothesis.byte_end}")
                else:
                    break
            else:
                break
        
        return validation_result
    
    def _calculate_confidence_summary(self, hypotheses: List[AttributeHypothesis]) -> Dict[str, int]:
        """Calculate confidence level summary."""
        summary = {'high': 0, 'medium': 0, 'low': 0}
        
        for hypothesis in hypotheses:
            if hypothesis.confidence >= 0.8:
                summary['high'] += 1
            elif hypothesis.confidence >= 0.5:
                summary['medium'] += 1
            else:
                summary['low'] += 1
        
        return summary
    
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.config.verbose_logging else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Suppress noisy loggers unless in debug mode
        if not self.config.verbose_logging:
            # Suppress segyio warnings
            logging.getLogger('segyio').setLevel(logging.ERROR)
            
            # Suppress urllib3 debug messages
            logging.getLogger('urllib3').setLevel(logging.WARNING)
            logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
            
            # Suppress requests debug messages
            logging.getLogger('requests').setLevel(logging.WARNING)
            logging.getLogger('requests.packages.urllib3').setLevel(logging.WARNING)
            
            # Suppress other common noisy loggers
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('httpcore').setLevel(logging.WARNING)    

    def _is_explicit_mapping(self, hypothesis: AttributeHypothesis) -> bool:
        """
        Check if a hypothesis represents an explicit byte location mapping from textual header.
        
        Args:
            hypothesis: The attribute hypothesis to check
            
        Returns:
            True if this appears to be an explicit mapping from textual header
        """
        # Check if reasoning contains explicit byte location indicators
        reasoning_lower = hypothesis.reasoning.lower() if hypothesis.reasoning else ""
        
        explicit_indicators = [
            'byte location', 'bytes', 'explicitly', 'header explicitly',
            'byte range', 'stated', 'documented', 'definition'
        ]
        
        # Check if reasoning suggests explicit documentation
        has_explicit_indicator = any(indicator in reasoning_lower for indicator in explicit_indicators)
        
        # Check if byte range follows standard patterns (like 21-24, 181-184, etc.)
        byte_range_size = hypothesis.byte_end - hypothesis.byte_start + 1
        is_standard_size = byte_range_size in [2, 4, 8]  # Standard SEGY data type sizes
        
        # Check if byte start follows common SEGY alignment
        is_aligned = hypothesis.byte_start % 2 == 1  # SEGY trace headers start at odd bytes
        
        # High confidence from LLM source suggests explicit documentation
        is_high_confidence_llm = hypothesis.confidence >= 0.85 and 'llm' in hypothesis.source
        
        return (has_explicit_indicator or is_high_confidence_llm) and is_standard_size and is_aligned
    
    def _should_skip_validation(self, hypothesis: AttributeHypothesis) -> bool:
        """
        Determine if validation should be skipped for a hypothesis.
        
        Args:
            hypothesis: The attribute hypothesis to check
            
        Returns:
            True if validation should be skipped
        """
        # Skip validation for very high confidence explicit mappings
        if hypothesis.confidence >= 0.9 and self._is_explicit_mapping(hypothesis):
            return True
        
        # Skip validation for standard SEGY locations with high confidence
        standard_locations = {
            (1, 4): 'trace_sequence_line',
            (5, 8): 'trace_sequence_file', 
            (9, 12): 'field_record',
            (13, 16): 'trace_number',
            (17, 20): 'energy_source_point',
            (21, 24): 'cdp_number',
            (73, 76): 'source_x',
            (77, 80): 'source_y',
            (81, 84): 'group_x',
            (85, 88): 'group_y',
            (181, 184): 'inline_3d',
            (185, 188): 'crossline_3d',
            (193, 196): 'cdp_x',
            (197, 200): 'cdp_y'
        }
        
        byte_location = (hypothesis.byte_start, hypothesis.byte_end)
        if byte_location in standard_locations and hypothesis.confidence >= 0.8:
            return True
        
        return False