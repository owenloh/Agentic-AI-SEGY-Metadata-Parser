"""
GeometricExtractor - Extracts geometric information from SEGY files.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from models.data_models import (
    AttributeHypothesis, GeometricInfo, ValidationStatus, 
    ConfidenceLevel, RevisionInfo
)
from core.segy_file_handler import SEGYFileHandler
from core.attribute_ontology import AttributeOntology


class GeometricExtractor:
    """Extracts geometric information (coordinates, angles, inline/crossline) from SEGY files"""
    
    def __init__(self, ontology: AttributeOntology):
        self.ontology = ontology
        
    def extract_geometric_info(
        self, 
        segy_handler: SEGYFileHandler,
        validated_attributes: List[AttributeHypothesis],
        revision_info: RevisionInfo,
        header_geometric_data: Optional[Any] = None
    ) -> GeometricInfo:
        """
        Extract comprehensive geometric information from SEGY file
        
        Args:
            segy_handler: SEGY file handler
            validated_attributes: List of validated attribute hypotheses
            revision_info: Detected SEGY revision information
            header_geometric_data: Enhanced geometric data from textual header parsing
            
        Returns:
            GeometricInfo object with extracted geometric data
        """
        # Enhanced extraction using header geometric data
        if header_geometric_data:
            # Use enhanced geometric data from textual header
            world_coordinates = self._extract_enhanced_world_coordinates(
                segy_handler, validated_attributes, header_geometric_data
            )
            
            azimuthal_angle = self._extract_enhanced_azimuthal_angle(
                header_geometric_data
            )
            
            inline_crossline = self._extract_enhanced_inline_crossline(
                segy_handler, validated_attributes, header_geometric_data
            )
            
            # Extract CRS info from header geometric data
            crs_info = self._extract_enhanced_crs_info(header_geometric_data)
            
            # Extract projection info from header geometric data
            projection_info = self._extract_enhanced_projection_info(header_geometric_data)
        else:
            # Fallback to original extraction methods
            world_coordinates = self._extract_world_coordinates(
                segy_handler, validated_attributes, revision_info
            )
            
            azimuthal_angle = self._extract_azimuthal_angle(
                segy_handler, validated_attributes, revision_info
            )
            
            inline_crossline = self._extract_inline_crossline(
                segy_handler, validated_attributes, revision_info
            )
            
            crs_info = self._extract_crs_info(segy_handler)
            projection_info = self._extract_projection_info(segy_handler)
        
        return GeometricInfo(
            world_coordinates=world_coordinates,
            azimuthal_angle=azimuthal_angle,
            inline_crossline=inline_crossline,
            coordinate_reference_system=crs_info,
            projection_info=projection_info
        )
    
    def _extract_world_coordinates(
        self,
        segy_handler: SEGYFileHandler,
        validated_attributes: List[AttributeHypothesis],
        revision_info: RevisionInfo
    ) -> Dict[str, AttributeHypothesis]:
        """Extract world coordinates (X, Y, Z) with cross-validation"""
        coordinates = {}
        
        # Define coordinate attribute mappings
        coordinate_mappings = {
            'X': ['source_x', 'receiver_x', 'cdp_x'],
            'Y': ['source_y', 'receiver_y', 'cdp_y'],
            'Z': ['source_z', 'receiver_z', 'cdp_z', 'elevation']
        }
        
        for coord_type, attribute_names in coordinate_mappings.items():
            best_hypothesis = None
            best_confidence = 0.0
            
            # Look for validated attributes first
            for attr in validated_attributes:
                if attr.attribute_name in attribute_names and attr.validation_status == ValidationStatus.PASSED:
                    if attr.confidence > best_confidence:
                        best_hypothesis = attr
                        best_confidence = attr.confidence
            
            # If no validated attribute found, try to create from standard locations
            if not best_hypothesis:
                best_hypothesis = self._create_standard_coordinate_hypothesis(
                    coordinate_mappings[coord_type][0],  # Use first option as default
                    revision_info,
                    segy_handler
                )
            
            if best_hypothesis:
                # Perform cross-validation with other coordinates
                if self._validate_coordinate_consistency(best_hypothesis, coordinates, segy_handler):
                    coordinates[coord_type] = best_hypothesis
        
        return coordinates
    
    def _extract_azimuthal_angle(
        self,
        segy_handler: SEGYFileHandler,
        validated_attributes: List[AttributeHypothesis],
        revision_info: RevisionInfo
    ) -> Optional[AttributeHypothesis]:
        """Extract azimuthal angle information"""
        azimuth_attributes = ['azimuth', 'source_azimuth', 'receiver_azimuth']
        
        # Look for validated azimuth attributes
        for attr in validated_attributes:
            if attr.attribute_name in azimuth_attributes and attr.validation_status == ValidationStatus.PASSED:
                return attr
        
        # Try to detect azimuth from standard locations or patterns
        return self._detect_azimuth_from_patterns(segy_handler, revision_info)
    
    def _extract_inline_crossline(
        self,
        segy_handler: SEGYFileHandler,
        validated_attributes: List[AttributeHypothesis],
        revision_info: RevisionInfo
    ) -> Dict[str, AttributeHypothesis]:
        """Extract inline/crossline mapping with cross-validation"""
        inline_crossline = {}
        
        # Look for validated inline/crossline attributes
        inline_attr = None
        crossline_attr = None
        
        for attr in validated_attributes:
            if attr.attribute_name in ['inline_number', 'inline_3d'] and attr.validation_status == ValidationStatus.PASSED:
                inline_attr = attr
            elif attr.attribute_name in ['crossline_number', 'crossline_3d'] and attr.validation_status == ValidationStatus.PASSED:
                crossline_attr = attr
        
        # If not found in validated attributes, create from standard locations
        if not inline_attr:
            inline_attr = self._create_standard_coordinate_hypothesis(
                'inline_number', revision_info, segy_handler
            )
        
        if not crossline_attr:
            crossline_attr = self._create_standard_coordinate_hypothesis(
                'crossline_number', revision_info, segy_handler
            )
        
        # Cross-validate inline/crossline consistency
        if inline_attr and crossline_attr:
            if self._validate_inline_crossline_consistency(inline_attr, crossline_attr, segy_handler):
                inline_crossline['inline'] = inline_attr
                inline_crossline['crossline'] = crossline_attr
        
        return inline_crossline
    
    def _create_standard_coordinate_hypothesis(
        self,
        attribute_name: str,
        revision_info: RevisionInfo,
        segy_handler: SEGYFileHandler
    ) -> Optional[AttributeHypothesis]:
        """Create hypothesis from standard SEGY locations"""
        try:
            schema = self.ontology.get_attribute_schema(attribute_name)
            if not schema:
                return None
            
            # Get standard locations for the detected revision
            revision = revision_info.revision if revision_info.revision != 'None' else '1.0'
            standard_locations = schema.standard_locations.get(revision, [])
            
            if not standard_locations:
                return None
            
            # Use the first standard location
            byte_start, byte_end = standard_locations[0]
            
            # Create hypothesis with medium confidence (since it's from standard location)
            hypothesis = AttributeHypothesis(
                attribute_name=attribute_name,
                byte_start=byte_start,
                byte_end=byte_end,
                confidence=0.6,  # Medium confidence for standard locations
                data_type=schema.expected_types[0] if schema.expected_types else 'int32',
                source='standard_location',
                reasoning=f'Standard SEGY location for {attribute_name} in revision {revision}',
                validation_status=ValidationStatus.NOT_VALIDATED
            )
            
            return hypothesis
            
        except Exception as e:
            print(f"Error creating standard coordinate hypothesis for {attribute_name}: {e}")
            return None
    
    def _validate_coordinate_consistency(
        self,
        new_coordinate: AttributeHypothesis,
        existing_coordinates: Dict[str, AttributeHypothesis],
        segy_handler: SEGYFileHandler
    ) -> bool:
        """Validate that coordinates are consistent with each other"""
        if not existing_coordinates:
            return True
        
        try:
            # Extract sample data for the new coordinate
            sample_indices = segy_handler.get_sample_trace_indices(50)
            new_data = segy_handler.get_trace_header_sample(
                sample_indices, (new_coordinate.byte_start, new_coordinate.byte_end)
            )
            
            if len(new_data) == 0:
                return False
            
            # Check consistency with existing coordinates
            for coord_type, existing_coord in existing_coordinates.items():
                existing_data = segy_handler.get_trace_header_sample(
                    sample_indices, (existing_coord.byte_start, existing_coord.byte_end)
                )
                
                if len(existing_data) == 0:
                    continue
                
                # Check if coordinates are in reasonable ranges relative to each other
                if not self._check_coordinate_range_consistency(new_data, existing_data):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validating coordinate consistency: {e}")
            return False
    
    def _check_coordinate_range_consistency(self, coord1_data: np.ndarray, coord2_data: np.ndarray) -> bool:
        """Check if two coordinate datasets have consistent ranges"""
        try:
            # Remove zeros and invalid values
            coord1_valid = coord1_data[coord1_data != 0]
            coord2_valid = coord2_data[coord2_data != 0]
            
            if len(coord1_valid) == 0 or len(coord2_valid) == 0:
                return True  # Can't validate, assume consistent
            
            # Check if ranges are in similar orders of magnitude
            coord1_range = np.max(coord1_valid) - np.min(coord1_valid)
            coord2_range = np.max(coord2_valid) - np.min(coord2_valid)
            
            if coord1_range == 0 or coord2_range == 0:
                return True
            
            # Ranges should be within 3 orders of magnitude of each other
            range_ratio = max(coord1_range, coord2_range) / min(coord1_range, coord2_range)
            return range_ratio < 1000
            
        except Exception:
            return True  # If we can't validate, assume consistent
    
    def _validate_inline_crossline_consistency(
        self,
        inline_attr: AttributeHypothesis,
        crossline_attr: AttributeHypothesis,
        segy_handler: SEGYFileHandler
    ) -> bool:
        """Validate inline/crossline consistency"""
        try:
            sample_indices = segy_handler.get_sample_trace_indices(100)
            
            inline_data = segy_handler.get_trace_header_sample(
                sample_indices, (inline_attr.byte_start, inline_attr.byte_end)
            )
            crossline_data = segy_handler.get_trace_header_sample(
                sample_indices, (crossline_attr.byte_start, crossline_attr.byte_end)
            )
            
            if len(inline_data) == 0 or len(crossline_data) == 0:
                return False
            
            # Remove invalid values
            inline_valid = inline_data[(inline_data > 0) & (inline_data < 1000000)]
            crossline_valid = crossline_data[(crossline_data > 0) & (crossline_data < 1000000)]
            
            if len(inline_valid) == 0 or len(crossline_valid) == 0:
                return False
            
            # Check if we have reasonable ranges for survey geometry
            inline_range = np.max(inline_valid) - np.min(inline_valid)
            crossline_range = np.max(crossline_valid) - np.min(crossline_valid)
            
            # Both should have some variation (not all the same value)
            return inline_range > 0 and crossline_range > 0
            
        except Exception as e:
            print(f"Error validating inline/crossline consistency: {e}")
            return False
    
    def _detect_azimuth_from_patterns(
        self,
        segy_handler: SEGYFileHandler,
        revision_info: RevisionInfo
    ) -> Optional[AttributeHypothesis]:
        """Attempt to detect azimuthal angle from data patterns"""
        # This is a simplified implementation
        # In practice, you might analyze coordinate patterns to infer azimuth
        
        # Common azimuth byte locations to try
        azimuth_locations = [
            (89, 92),   # Common non-standard location
            (197, 200), # Another common location
            (201, 204)  # Alternative location
        ]
        
        for byte_start, byte_end in azimuth_locations:
            try:
                sample_indices = segy_handler.get_sample_trace_indices(50)
                data = segy_handler.get_trace_header_sample(sample_indices, (byte_start, byte_end))
                
                if len(data) > 0:
                    # Check if data looks like azimuth (0-360 degrees or 0-2π radians)
                    valid_data = data[data != 0]
                    if len(valid_data) > 0:
                        min_val, max_val = np.min(valid_data), np.max(valid_data)
                        
                        # Check for degree range (0-360)
                        if 0 <= min_val and max_val <= 360:
                            return AttributeHypothesis(
                                attribute_name='azimuth',
                                byte_start=byte_start,
                                byte_end=byte_end,
                                confidence=0.4,  # Low confidence for pattern detection
                                data_type='float32',
                                source='pattern_detection',
                                reasoning=f'Detected azimuth pattern at bytes {byte_start}-{byte_end}, range {min_val:.1f}-{max_val:.1f} degrees',
                                validation_status=ValidationStatus.NOT_VALIDATED
                            )
                        
                        # Check for radian range (0-2π ≈ 6.28)
                        elif 0 <= min_val and max_val <= 7:
                            return AttributeHypothesis(
                                attribute_name='azimuth',
                                byte_start=byte_start,
                                byte_end=byte_end,
                                confidence=0.4,
                                data_type='float32',
                                source='pattern_detection',
                                reasoning=f'Detected azimuth pattern at bytes {byte_start}-{byte_end}, range {min_val:.2f}-{max_val:.2f} radians',
                                validation_status=ValidationStatus.NOT_VALIDATED
                            )
            
            except Exception:
                continue
        
        return None
    
    def _extract_crs_info(self, segy_handler: SEGYFileHandler) -> Optional[str]:
        """Extract coordinate reference system information from textual headers"""
        try:
            textual_headers = segy_handler.get_textual_headers()
            
            # Look for CRS/projection keywords in textual headers
            crs_keywords = [
                'UTM', 'WGS84', 'NAD83', 'NAD27', 'EPSG', 'PROJ',
                'COORDINATE SYSTEM', 'PROJECTION', 'DATUM', 'ZONE'
            ]
            
            for header in textual_headers:
                header_upper = header.upper()
                for keyword in crs_keywords:
                    if keyword in header_upper:
                        # Extract the line containing CRS information
                        lines = header.split('\n')
                        for line in lines:
                            if keyword in line.upper():
                                return line.strip()
            
            return None
            
        except Exception as e:
            print(f"Error extracting CRS info: {e}")
            return None
    
    def _extract_projection_info(self, segy_handler: SEGYFileHandler) -> Optional[str]:
        """Extract projection information from textual headers"""
        try:
            textual_headers = segy_handler.get_textual_headers()
            
            # Look for projection-specific information
            projection_keywords = [
                'TRANSVERSE MERCATOR', 'LAMBERT', 'ALBERS', 'STEREOGRAPHIC',
                'MERCATOR', 'GEOGRAPHIC', 'CARTESIAN', 'LOCAL'
            ]
            
            for header in textual_headers:
                header_upper = header.upper()
                for keyword in projection_keywords:
                    if keyword in header_upper:
                        lines = header.split('\n')
                        for line in lines:
                            if keyword in line.upper():
                                return line.strip()
            
            return None
            
        except Exception as e:
            print(f"Error extracting projection info: {e}")
            return None    

    def _extract_enhanced_world_coordinates(self, segy_handler, validated_attributes, header_geometric_data):
        """Extract world coordinates using enhanced geometric data."""
        world_coordinates = {}
        
        # First, try to get coordinates from validated attributes (trace header mappings)
        for attr in validated_attributes:
            attr_name_lower = attr.attribute_name.lower()
            
            if any(x_term in attr_name_lower for x_term in ['easting', 'x', 'source_x', 'group_x']):
                world_coordinates['X'] = attr
            elif any(y_term in attr_name_lower for y_term in ['northing', 'y', 'source_y', 'group_y']):
                world_coordinates['Y'] = attr
            elif any(z_term in attr_name_lower for z_term in ['elevation', 'z', 'depth']):
                world_coordinates['Z'] = attr
        
        # If we have corner points from header, we can infer coordinate system
        if header_geometric_data.corner_points and len(header_geometric_data.corner_points) > 0:
            # Corner points provide validation that we have the right coordinate system
            corner_point = header_geometric_data.corner_points[0]
            
            # If we don't have X/Y from trace headers, create hypotheses based on standard locations
            if 'X' not in world_coordinates and corner_point.get('easting'):
                # Standard SEGY locations for X coordinates
                for standard_x_byte in [73, 81, 181, 193]:  # source_x, group_x, cdp_x, inline_x
                    x_attr = AttributeHypothesis(
                        attribute_name="X Coordinate (from corner points)",
                        byte_start=standard_x_byte,
                        byte_end=standard_x_byte + 3,
                        data_type='int32',
                        confidence=0.8,
                        source='enhanced_geometric',
                        reasoning=f"Inferred from corner points with easting {corner_point['easting']}"
                    )
                    world_coordinates['X'] = x_attr
                    break
            
            if 'Y' not in world_coordinates and corner_point.get('northing'):
                # Standard SEGY locations for Y coordinates
                for standard_y_byte in [77, 85, 185, 197]:  # source_y, group_y, cdp_y, inline_y
                    y_attr = AttributeHypothesis(
                        attribute_name="Y Coordinate (from corner points)",
                        byte_start=standard_y_byte,
                        byte_end=standard_y_byte + 3,
                        data_type='int32',
                        confidence=0.8,
                        source='enhanced_geometric',
                        reasoning=f"Inferred from corner points with northing {corner_point['northing']}"
                    )
                    world_coordinates['Y'] = y_attr
                    break
        
        return world_coordinates
    
    def _extract_enhanced_azimuthal_angle(self, header_geometric_data):
        """Extract azimuthal angle from enhanced geometric data."""
        if header_geometric_data.survey_geometry:
            line_direction = header_geometric_data.survey_geometry.get('line_direction')
            if line_direction is not None:
                # Create an attribute hypothesis for azimuthal angle
                return AttributeHypothesis(
                    attribute_name="Azimuthal Angle",
                    byte_start=0,  # Not from trace header
                    byte_end=0,
                    data_type='float32',
                    confidence=0.9,
                    source='enhanced_geometric',
                    reasoning=f"Line direction from survey geometry: {line_direction}°"
                )
        
        return None
    
    def _extract_enhanced_inline_crossline(self, segy_handler, validated_attributes, header_geometric_data):
        """Extract inline/crossline mapping using enhanced geometric data."""
        inline_crossline = {}
        
        # Get inline/crossline from validated attributes
        for attr in validated_attributes:
            attr_name_lower = attr.attribute_name.lower()
            
            if any(inline_term in attr_name_lower for inline_term in ['inline', 'line', '3d line']):
                inline_crossline['inline'] = attr
            elif any(xline_term in attr_name_lower for xline_term in ['crossline', 'xline', 'cdp', '3d cdp']):
                inline_crossline['crossline'] = attr
        
        # Enhance with survey geometry information
        if header_geometric_data.survey_geometry:
            survey_geom = header_geometric_data.survey_geometry
            
            # Add survey geometry metadata to existing attributes
            if 'inline' in inline_crossline:
                inline_attr = inline_crossline['inline']
                inline_attr.reasoning += f" | Survey: origin={survey_geom.get('line_origin')}, inc={survey_geom.get('line_increment')}, dir={survey_geom.get('line_direction')}°"
            
            if 'crossline' in inline_crossline:
                xline_attr = inline_crossline['crossline']
                xline_attr.reasoning += f" | Survey: origin={survey_geom.get('cdp_origin')}, inc={survey_geom.get('cdp_increment')}, dir={survey_geom.get('cdp_direction')}°"
        
        return inline_crossline
    
    def _extract_enhanced_crs_info(self, header_geometric_data):
        """Extract coordinate reference system info from enhanced geometric data."""
        if header_geometric_data.coordinate_system:
            coord_sys = header_geometric_data.coordinate_system
            datum = coord_sys.get('datum', '')
            projection = coord_sys.get('projection', '')
            
            if datum or projection:
                return f"{datum} {projection}".strip()
        
        return None
    
    def _extract_enhanced_projection_info(self, header_geometric_data):
        """Extract projection information from enhanced geometric data."""
        if header_geometric_data.coordinate_system:
            coord_sys = header_geometric_data.coordinate_system
            projection = coord_sys.get('projection', '')
            zone = coord_sys.get('zone', '')
            
            if projection:
                proj_info = projection
                if zone:
                    proj_info += f" Zone {zone}"
                return proj_info
        
        # Also check for grid origin information
        if header_geometric_data.grid_origin:
            grid_origin = header_geometric_data.grid_origin
            easting = grid_origin.get('easting')
            northing = grid_origin.get('northing')
            
            if easting and northing:
                return f"Grid Origin: {easting}E, {northing}N"
        
        return None