"""
Fallback strategy manager for SEGY header parsing.

This module provides fallback strategies when LLM-based parsing fails or needs
supplementation, including standard byte locations and binary header analysis.
"""

import logging
import struct
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from core.llm_header_parser import AttributeHypothesis, RevisionInfo
from core.attribute_ontology import AttributeOntology
from core.segy_file_handler import SEGYFileHandler


class FallbackStrategyManager:
    """
    Manages fallback strategies for SEGY header parsing when LLM approaches fail.
    """
    
    def __init__(self, ontology: AttributeOntology):
        self.ontology = ontology
        self.logger = logging.getLogger(__name__)
        
        # Standard SEGY byte locations (from SEG-Y rev 1 and 2 specifications)
        self.STANDARD_LOCATIONS = {
            # Binary header locations (bytes 3201-3600)
            'job_id': (3201, 3204, 'int32'),
            'line_number': (3205, 3208, 'int32'),
            'reel_number': (3209, 3212, 'int32'),
            'traces_per_ensemble': (3213, 3214, 'int16'),
            'auxiliary_traces_per_ensemble': (3215, 3216, 'int16'),
            'sample_interval': (3217, 3218, 'int16'),
            'sample_interval_original': (3219, 3220, 'int16'),
            'samples_per_trace': (3221, 3222, 'int16'),
            'samples_per_trace_original': (3223, 3224, 'int16'),
            'data_sample_format': (3225, 3226, 'int16'),
            'ensemble_fold': (3227, 3228, 'int16'),
            'trace_sorting': (3229, 3230, 'int16'),
            'vertical_sum_code': (3231, 3232, 'int16'),
            'sweep_frequency_start': (3233, 3234, 'int16'),
            'sweep_frequency_end': (3235, 3236, 'int16'),
            'sweep_length': (3237, 3238, 'int16'),
            'sweep_type': (3239, 3240, 'int16'),
            'trace_number_sweep_channel': (3241, 3242, 'int16'),
            'sweep_trace_taper_start': (3243, 3244, 'int16'),
            'sweep_trace_taper_end': (3245, 3246, 'int16'),
            'taper_type': (3247, 3248, 'int16'),
            'correlated_data_traces': (3249, 3250, 'int16'),
            'binary_gain_recovered': (3251, 3252, 'int16'),
            'amplitude_recovery_method': (3253, 3254, 'int16'),
            'measurement_system': (3255, 3256, 'int16'),
            'impulse_signal_polarity': (3257, 3258, 'int16'),
            'vibratory_polarity_code': (3259, 3260, 'int16'),
            'segy_revision': (3501, 3502, 'int16'),
            'fixed_length_trace_flag': (3503, 3504, 'int16'),
            'extended_textual_headers': (3505, 3506, 'int16'),
            
            # Common trace header locations (bytes 1-240)
            'trace_sequence_line': (1, 4, 'int32'),
            'trace_sequence_file': (5, 8, 'int32'),
            'original_field_record': (9, 12, 'int32'),
            'trace_number_field_record': (13, 16, 'int32'),
            'energy_source_point': (17, 20, 'int32'),
            'ensemble_number': (21, 24, 'int32'),
            'trace_number_ensemble': (25, 28, 'int32'),
            'trace_identification_code': (29, 30, 'int16'),
            'vertical_sum_traces': (31, 32, 'int16'),
            'horizontal_stack_traces': (33, 34, 'int16'),
            'data_use': (35, 36, 'int16'),
            'distance_source_receiver': (37, 40, 'int32'),
            'receiver_group_elevation': (41, 44, 'int32'),
            'surface_elevation_source': (45, 48, 'int32'),
            'source_depth': (49, 52, 'int32'),
            'datum_elevation_receiver': (53, 56, 'int32'),
            'datum_elevation_source': (57, 60, 'int32'),
            'water_depth_source': (61, 64, 'int32'),
            'water_depth_receiver': (65, 68, 'int32'),
            'elevation_scalar': (69, 70, 'int16'),
            'coordinate_scalar': (71, 72, 'int16'),
            'source_x': (73, 76, 'int32'),
            'source_y': (77, 80, 'int32'),
            'receiver_x': (81, 84, 'int32'),
            'receiver_y': (85, 88, 'int32'),
            'coordinate_units': (89, 90, 'int16'),
            'weathering_velocity': (91, 92, 'int16'),
            'subweathering_velocity': (93, 94, 'int16'),
            'uphole_time_source': (95, 96, 'int16'),
            'uphole_time_receiver': (97, 98, 'int16'),
            'source_static_correction': (99, 100, 'int16'),
            'receiver_static_correction': (101, 102, 'int16'),
            'total_static_applied': (103, 104, 'int16'),
            'lag_time_a': (105, 106, 'int16'),
            'lag_time_b': (107, 108, 'int16'),
            'delay_recording_time': (109, 110, 'int16'),
            'mute_time_start': (111, 112, 'int16'),
            'mute_time_end': (113, 114, 'int16'),
            'samples_this_trace': (115, 116, 'int16'),
            'sample_interval_this_trace': (117, 118, 'int16'),
            'gain_type': (119, 120, 'int16'),
            'instrument_gain_constant': (121, 122, 'int16'),
            'instrument_early_gain': (123, 124, 'int16'),
            'correlated': (125, 126, 'int16'),
            'sweep_frequency_start_trace': (127, 128, 'int16'),
            'sweep_frequency_end_trace': (129, 130, 'int16'),
            'sweep_length_trace': (131, 132, 'int16'),
            'sweep_type_trace': (133, 134, 'int16'),
            'sweep_trace_taper_start_trace': (135, 136, 'int16'),
            'sweep_trace_taper_end_trace': (137, 138, 'int16'),
            'taper_type_trace': (139, 140, 'int16'),
            'alias_filter_frequency': (141, 142, 'int16'),
            'alias_filter_slope': (143, 144, 'int16'),
            'notch_filter_frequency': (145, 146, 'int16'),
            'notch_filter_slope': (147, 148, 'int16'),
            'low_cut_frequency': (149, 150, 'int16'),
            'high_cut_frequency': (151, 152, 'int16'),
            'low_cut_slope': (153, 154, 'int16'),
            'high_cut_slope': (155, 156, 'int16'),
            'year': (157, 158, 'int16'),
            'day_of_year': (159, 160, 'int16'),
            'hour': (161, 162, 'int16'),
            'minute': (163, 164, 'int16'),
            'second': (165, 166, 'int16'),
            'time_basis_code': (167, 168, 'int16'),
            'trace_weighting_factor': (169, 170, 'int16'),
            'geophone_group_roll_position': (171, 172, 'int16'),
            'geophone_group_first_trace': (173, 174, 'int16'),
            'geophone_group_last_trace': (175, 176, 'int16'),
            'gap_size': (177, 178, 'int16'),
            'over_travel': (179, 180, 'int16'),
            'x_coordinate_ensemble': (181, 184, 'int32'),
            'y_coordinate_ensemble': (185, 188, 'int32'),
            'inline_number': (189, 192, 'int32'),
            'crossline_number': (193, 196, 'int32'),
            'shotpoint_number': (197, 200, 'int32'),
            'shotpoint_scalar': (201, 202, 'int16'),
            'trace_value_measurement_unit': (203, 204, 'int16'),
            'transduction_constant': (205, 210, 'int48'),  # Special case
            'transduction_units': (211, 212, 'int16'),
            'device_identifier': (213, 214, 'int16'),
            'time_scalar': (215, 216, 'int16'),
            'source_type': (217, 218, 'int16'),
            'source_energy_direction': (219, 224, 'int48'),  # Special case
            'source_measurement': (225, 230, 'int48'),  # Special case
            'source_measurement_unit': (231, 232, 'int16')
        }
    
    def generate_fallback_hypotheses(self, 
                                   segy_file_path: Path, 
                                   revision_info: RevisionInfo,
                                   existing_hypotheses: List[AttributeHypothesis]) -> List[AttributeHypothesis]:
        """
        Generate fallback hypotheses using standard locations and binary analysis.
        
        Args:
            segy_file_path: Path to SEGY file
            revision_info: Detected revision information
            existing_hypotheses: Already generated hypotheses to avoid duplicates
            
        Returns:
            List of fallback AttributeHypothesis objects
        """
        fallback_hypotheses = []
        
        # Get existing attribute names to avoid duplicates
        existing_attrs = {h.attribute_name.lower() for h in existing_hypotheses}
        
        # 1. Standard location fallbacks
        standard_hypotheses = self._generate_standard_location_hypotheses(revision_info, existing_attrs)
        fallback_hypotheses.extend(standard_hypotheses)
        
        # 2. Binary header analysis
        binary_hypotheses = self._analyze_binary_header(segy_file_path, existing_attrs)
        fallback_hypotheses.extend(binary_hypotheses)
        
        # 3. Pattern-based detection in textual header
        pattern_hypotheses = self._pattern_based_detection(segy_file_path, existing_attrs)
        fallback_hypotheses.extend(pattern_hypotheses)
        
        return fallback_hypotheses
    
    def _generate_standard_location_hypotheses(self, 
                                             revision_info: RevisionInfo, 
                                             existing_attrs: set) -> List[AttributeHypothesis]:
        """Generate hypotheses based on standard SEGY locations."""
        hypotheses = []
        
        # Filter standard locations based on revision
        relevant_locations = self._filter_by_revision(self.STANDARD_LOCATIONS, revision_info.revision)
        
        for attr_name, (start, end, data_type) in relevant_locations.items():
            if attr_name.lower() not in existing_attrs:
                hypothesis = AttributeHypothesis(
                    attribute_name=attr_name,
                    byte_start=start,
                    byte_end=end,
                    data_type=data_type,
                    confidence=0.6,  # Medium confidence for standard locations
                    source='fallback_standard',
                    reasoning=f"Standard SEGY {revision_info.revision} location for {attr_name}"
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _filter_by_revision(self, locations: Dict[str, Tuple], revision: str) -> Dict[str, Tuple]:
        """Filter standard locations based on SEGY revision."""
        if revision in ['None', '0']:
            # Original SEGY format - exclude revision 1+ specific fields
            excluded = {'segy_revision', 'fixed_length_trace_flag', 'extended_textual_headers',
                       'inline_number', 'crossline_number', 'shotpoint_number'}
            return {k: v for k, v in locations.items() if k not in excluded}
        
        elif revision in ['1', '1.0']:
            # SEGY revision 1 - include most fields
            excluded = {'extended_textual_headers'}  # Rev 2+ only
            return {k: v for k, v in locations.items() if k not in excluded}
        
        else:
            # SEGY revision 2+ - include all fields
            return locations
    
    def _analyze_binary_header(self, segy_file_path: Path, existing_attrs: set) -> List[AttributeHypothesis]:
        """Analyze binary header to detect actual values and infer attributes."""
        hypotheses = []
        
        try:
            segy_handler = SEGYFileHandler(segy_file_path)
            if not segy_handler.open_file():
                return hypotheses
            
            binary_header_dict = segy_handler.get_binary_header()
            segy_handler.close_file()
            
            if not binary_header_dict:
                return hypotheses
            
            # Convert dict to bytes for analysis (this is a simplified approach)
            # In practice, we'd need to access the raw binary header bytes
            binary_header = b''  # Placeholder - would need actual binary data
            
            # Analyze key binary header fields using the parsed dictionary
            binary_analyses = [
                self._analyze_sample_interval_dict(binary_header_dict),
                self._analyze_samples_per_trace_dict(binary_header_dict),
                self._analyze_data_format_dict(binary_header_dict),
                self._analyze_revision_number_dict(binary_header_dict),
                self._analyze_trace_sorting_dict(binary_header_dict)
            ]
            
            for analysis in binary_analyses:
                if analysis and analysis.attribute_name.lower() not in existing_attrs:
                    hypotheses.append(analysis)
            
        except Exception as e:
            self.logger.warning(f"Binary header analysis failed: {e}")
        
        return hypotheses
    
    def _analyze_sample_interval_dict(self, binary_header_dict: Dict[str, Any]) -> Optional[AttributeHypothesis]:
        """Analyze sample interval from binary header dictionary."""
        try:
            sample_interval = binary_header_dict.get('sample_interval', 0)
            
            if sample_interval > 0 and sample_interval < 100000:  # Reasonable range
                return AttributeHypothesis(
                    attribute_name='sample_interval',
                    byte_start=3217,
                    byte_end=3218,
                    data_type='int16',
                    confidence=0.8,
                    source='binary_analysis',
                    reasoning=f'Detected sample interval: {sample_interval} microseconds'
                )
        except Exception:
            pass
        return None
    
    def _analyze_sample_interval(self, binary_header: bytes) -> Optional[AttributeHypothesis]:
        """Analyze sample interval from binary header."""
        try:
            # Sample interval is at bytes 3217-3218 (0-indexed: 16-17 in binary header)
            if len(binary_header) >= 18:
                sample_interval = struct.unpack('>H', binary_header[16:18])[0]  # Big-endian uint16
                
                if sample_interval > 0 and sample_interval < 100000:  # Reasonable range
                    return AttributeHypothesis(
                        attribute_name='sample_interval',
                        byte_start=3217,
                        byte_end=3218,
                        data_type='int16',
                        confidence=0.8,
                        source='binary_analysis',
                        reasoning=f'Detected sample interval: {sample_interval} microseconds'
                    )
        except Exception:
            pass
        return None
    
    def _analyze_samples_per_trace_dict(self, binary_header_dict: Dict[str, Any]) -> Optional[AttributeHypothesis]:
        """Analyze samples per trace from binary header dictionary."""
        try:
            samples = binary_header_dict.get('samples_per_trace', 0)
            
            if samples > 0 and samples < 100000:  # Reasonable range
                return AttributeHypothesis(
                    attribute_name='samples_per_trace',
                    byte_start=3221,
                    byte_end=3222,
                    data_type='int16',
                    confidence=0.8,
                    source='binary_analysis',
                    reasoning=f'Detected samples per trace: {samples}'
                )
        except Exception:
            pass
        return None
    
    def _analyze_data_format_dict(self, binary_header_dict: Dict[str, Any]) -> Optional[AttributeHypothesis]:
        """Analyze data sample format from binary header dictionary."""
        try:
            format_code = binary_header_dict.get('data_sample_format', 0)
            
            format_names = {
                1: 'IBM floating point',
                2: '32-bit integer',
                3: '16-bit integer',
                4: 'Fixed point with gain',
                5: 'IEEE floating point',
                8: '8-bit integer'
            }
            
            if format_code in format_names:
                return AttributeHypothesis(
                    attribute_name='data_sample_format',
                    byte_start=3225,
                    byte_end=3226,
                    data_type='int16',
                    confidence=0.9,
                    source='binary_analysis',
                    reasoning=f'Detected data format: {format_code} ({format_names[format_code]})'
                )
        except Exception:
            pass
        return None
    
    def _analyze_revision_number_dict(self, binary_header_dict: Dict[str, Any]) -> Optional[AttributeHypothesis]:
        """Analyze SEGY revision number from binary header dictionary."""
        try:
            revision = binary_header_dict.get('segy_revision', 0)
            
            if revision in [0, 1, 2]:  # Valid revision numbers
                return AttributeHypothesis(
                    attribute_name='segy_revision',
                    byte_start=3501,
                    byte_end=3502,
                    data_type='int16',
                    confidence=0.95,
                    source='binary_analysis',
                    reasoning=f'Detected SEGY revision: {revision}'
                )
        except Exception:
            pass
        return None
    
    def _analyze_trace_sorting_dict(self, binary_header_dict: Dict[str, Any]) -> Optional[AttributeHypothesis]:
        """Analyze trace sorting code from binary header dictionary."""
        try:
            sorting = binary_header_dict.get('trace_sorting_code', 0)
            
            sorting_names = {
                -1: 'Other',
                0: 'Unknown',
                1: 'As recorded',
                2: 'CDP ensemble',
                3: 'Single fold continuous profile',
                4: 'Horizontally stacked'
            }
            
            if sorting in sorting_names:
                return AttributeHypothesis(
                    attribute_name='trace_sorting',
                    byte_start=3229,
                    byte_end=3230,
                    data_type='int16',
                    confidence=0.7,
                    source='binary_analysis',
                    reasoning=f'Detected trace sorting: {sorting} ({sorting_names[sorting]})'
                )
        except Exception:
            pass
        return None
    
    def _analyze_samples_per_trace(self, binary_header: bytes) -> Optional[AttributeHypothesis]:
        """Analyze samples per trace from binary header."""
        try:
            # Samples per trace is at bytes 3221-3222 (0-indexed: 20-21 in binary header)
            if len(binary_header) >= 22:
                samples = struct.unpack('>H', binary_header[20:22])[0]  # Big-endian uint16
                
                if samples > 0 and samples < 100000:  # Reasonable range
                    return AttributeHypothesis(
                        attribute_name='samples_per_trace',
                        byte_start=3221,
                        byte_end=3222,
                        data_type='int16',
                        confidence=0.8,
                        source='binary_analysis',
                        reasoning=f'Detected samples per trace: {samples}'
                    )
        except Exception:
            pass
        return None
    
    def _analyze_data_format(self, binary_header: bytes) -> Optional[AttributeHypothesis]:
        """Analyze data sample format from binary header."""
        try:
            # Data sample format is at bytes 3225-3226 (0-indexed: 24-25 in binary header)
            if len(binary_header) >= 26:
                format_code = struct.unpack('>H', binary_header[24:26])[0]  # Big-endian uint16
                
                format_names = {
                    1: 'IBM floating point',
                    2: '32-bit integer',
                    3: '16-bit integer',
                    4: 'Fixed point with gain',
                    5: 'IEEE floating point',
                    8: '8-bit integer'
                }
                
                if format_code in format_names:
                    return AttributeHypothesis(
                        attribute_name='data_sample_format',
                        byte_start=3225,
                        byte_end=3226,
                        data_type='int16',
                        confidence=0.9,
                        source='binary_analysis',
                        reasoning=f'Detected data format: {format_code} ({format_names[format_code]})'
                    )
        except Exception:
            pass
        return None
    
    def _analyze_revision_number(self, binary_header: bytes) -> Optional[AttributeHypothesis]:
        """Analyze SEGY revision number from binary header."""
        try:
            # SEGY revision is at bytes 3501-3502 (0-indexed: 300-301 in binary header)
            if len(binary_header) >= 302:
                revision = struct.unpack('>H', binary_header[300:302])[0]  # Big-endian uint16
                
                if revision in [0, 1, 2]:  # Valid revision numbers
                    return AttributeHypothesis(
                        attribute_name='segy_revision',
                        byte_start=3501,
                        byte_end=3502,
                        data_type='int16',
                        confidence=0.95,
                        source='binary_analysis',
                        reasoning=f'Detected SEGY revision: {revision}'
                    )
        except Exception:
            pass
        return None
    
    def _analyze_trace_sorting(self, binary_header: bytes) -> Optional[AttributeHypothesis]:
        """Analyze trace sorting code from binary header."""
        try:
            # Trace sorting is at bytes 3229-3230 (0-indexed: 28-29 in binary header)
            if len(binary_header) >= 30:
                sorting = struct.unpack('>H', binary_header[28:30])[0]  # Big-endian uint16
                
                sorting_names = {
                    -1: 'Other',
                    0: 'Unknown',
                    1: 'As recorded',
                    2: 'CDP ensemble',
                    3: 'Single fold continuous profile',
                    4: 'Horizontally stacked'
                }
                
                if sorting in sorting_names:
                    return AttributeHypothesis(
                        attribute_name='trace_sorting',
                        byte_start=3229,
                        byte_end=3230,
                        data_type='int16',
                        confidence=0.7,
                        source='binary_analysis',
                        reasoning=f'Detected trace sorting: {sorting} ({sorting_names[sorting]})'
                    )
        except Exception:
            pass
        return None
    
    def _pattern_based_detection(self, segy_file_path: Path, existing_attrs: set) -> List[AttributeHypothesis]:
        """Use pattern matching to detect attributes in textual header."""
        hypotheses = []
        
        try:
            segy_handler = SEGYFileHandler(segy_file_path)
            if not segy_handler.open_file():
                return hypotheses
            
            textual_headers = segy_handler.get_textual_headers()
            textual_header = '\n'.join(textual_headers) if textual_headers else ""
            segy_handler.close_file()
            
            if not textual_header:
                return hypotheses
            
            # Common patterns for attribute detection
            patterns = [
                (r'inline\s+(?:number|num).*?(?:byte|location).*?(\d+)[-–](\d+)', 'inline_number'),
                (r'crossline\s+(?:number|num).*?(?:byte|location).*?(\d+)[-–](\d+)', 'crossline_number'),
                (r'x\s+coord.*?(?:byte|location).*?(\d+)[-–](\d+)', 'source_x'),
                (r'y\s+coord.*?(?:byte|location).*?(\d+)[-–](\d+)', 'source_y'),
                (r'shotpoint.*?(?:byte|location).*?(\d+)[-–](\d+)', 'shotpoint_number'),
                (r'trace\s+(?:sequence|number).*?(?:byte|location).*?(\d+)[-–](\d+)', 'trace_sequence_line'),
                (r'elevation.*?(?:byte|location).*?(\d+)[-–](\d+)', 'receiver_group_elevation'),
                (r'sample\s+interval.*?(?:byte|location).*?(\d+)[-–](\d+)', 'sample_interval'),
            ]
            
            import re
            header_lower = textual_header.lower()
            
            for pattern, attr_name in patterns:
                if attr_name.lower() not in existing_attrs:
                    matches = re.finditer(pattern, header_lower, re.IGNORECASE)
                    
                    for match in matches:
                        try:
                            start_byte = int(match.group(1))
                            end_byte = int(match.group(2))
                            
                            hypothesis = AttributeHypothesis(
                                attribute_name=attr_name,
                                byte_start=start_byte,
                                byte_end=end_byte,
                                data_type='int32',  # Default assumption
                                confidence=0.5,
                                source='fallback_pattern',
                                reasoning=f'Pattern match: {match.group(0)}'
                            )
                            hypotheses.append(hypothesis)
                            break  # Only take first match per pattern
                            
                        except (ValueError, IndexError):
                            continue
            
        except Exception as e:
            self.logger.warning(f"Pattern-based detection failed: {e}")
        
        return hypotheses
    
    def enhance_hypotheses_with_context(self, 
                                      hypotheses: List[AttributeHypothesis], 
                                      segy_file_path: Path) -> List[AttributeHypothesis]:
        """Enhance hypotheses with additional context from file analysis."""
        enhanced = []
        
        for hypothesis in hypotheses:
            enhanced_hyp = self._enhance_single_hypothesis(hypothesis, segy_file_path)
            enhanced.append(enhanced_hyp)
        
        return enhanced
    
    def _enhance_single_hypothesis(self, 
                                 hypothesis: AttributeHypothesis, 
                                 segy_file_path: Path) -> AttributeHypothesis:
        """Enhance a single hypothesis with additional context."""
        # Create a copy to avoid modifying the original
        enhanced = AttributeHypothesis(
            attribute_name=hypothesis.attribute_name,
            byte_start=hypothesis.byte_start,
            byte_end=hypothesis.byte_end,
            data_type=hypothesis.data_type,
            confidence=hypothesis.confidence,
            source=hypothesis.source,
            reasoning=hypothesis.reasoning,
            validation_status=hypothesis.validation_status
        )
        
        # Check if byte range aligns with standard data type sizes
        byte_range_size = hypothesis.byte_end - hypothesis.byte_start + 1
        
        if hypothesis.data_type == 'int16' and byte_range_size != 2:
            enhanced.confidence *= 0.8
            enhanced.reasoning += "; Byte range size doesn't match int16"
        elif hypothesis.data_type == 'int32' and byte_range_size != 4:
            enhanced.confidence *= 0.8
            enhanced.reasoning += "; Byte range size doesn't match int32"
        elif hypothesis.data_type == 'float32' and byte_range_size != 4:
            enhanced.confidence *= 0.8
            enhanced.reasoning += "; Byte range size doesn't match float32"
        
        # Boost confidence for well-known standard locations
        if (hypothesis.byte_start, hypothesis.byte_end) in [(73, 76), (77, 80), (189, 192), (193, 196)]:
            enhanced.confidence = min(enhanced.confidence * 1.1, 1.0)
            enhanced.reasoning += "; Standard SEGY location"
        
        return enhanced