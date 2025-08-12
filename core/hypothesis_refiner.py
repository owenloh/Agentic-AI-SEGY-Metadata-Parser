"""
HypothesisRefiner - Refines byte location hypotheses when validation fails.

This module implements strategies for refining failed hypotheses through
byte alignment, data type alternatives, and location exploration.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from models.data_models import AttributeHypothesis, RefinementSuggestion, ValidationStatus, ValidationResult
from core.validation_llm import ValidationDecision
from core.llm_provider import LLMFactory
from core.attribute_ontology import AttributeOntology


class HypothesisRefiner:
    """Refines byte location hypotheses when validation fails"""
    
    def __init__(self, llm_factory: LLMFactory, ontology: AttributeOntology):
        self.llm_factory = llm_factory
        self.ontology = ontology
        # Standard SEGY byte alignments
        self.byte_alignments = [1, 2, 4, 8]
        
        # Common data type alternatives
        self.data_type_alternatives = {
            'int16': ['int32', 'float32', 'uint16'],
            'int32': ['float32', 'int16', 'uint32'],
            'float32': ['int32', 'float64', 'int16'],
            'float64': ['float32', 'int32', 'int64'],
            'uint16': ['int16', 'uint32', 'int32'],
            'uint32': ['int32', 'uint16', 'float32']
        }
        
        # Exploration radii for nearby location search
        self.exploration_offsets = [1, 2, 4, 8, 16, -1, -2, -4, -8, -16]
        
        # Standard SEGY trace header locations (revision 1)
        self.standard_locations = {
            'trace_sequence_line': (1, 4),
            'trace_sequence_file': (5, 8),
            'field_record': (9, 12),
            'trace_number': (13, 16),
            'energy_source_point': (17, 20),
            'cdp': (21, 24),
            'cdp_trace': (25, 28),
            'trace_identification_code': (29, 30),
            'source_x': (73, 76),
            'source_y': (77, 80),
            'group_x': (81, 84),
            'group_y': (85, 88),
            'cdp_x': (181, 184),
            'cdp_y': (185, 188),
            'inline_3d': (189, 192),
            'crossline_3d': (193, 196),
            'sample_interval': (117, 118),
            'samples_per_trace': (115, 116)
        }
    
    def refine_hypothesis(self, hypothesis: AttributeHypothesis, validation_result: ValidationResult) -> List[AttributeHypothesis]:
        """
        Refine a failed hypothesis by generating alternative hypotheses.
        
        Args:
            hypothesis: The original hypothesis that failed validation
            validation_result: The validation result containing failure details
            
        Returns:
            List of refined hypotheses to try
        """
        refined_hypotheses = []
        
        # Strategy 1: Try different data types
        for alt_data_type in self.data_type_alternatives.get(hypothesis.data_type, []):
            refined_hyp = AttributeHypothesis(
                attribute_name=hypothesis.attribute_name,
                byte_start=hypothesis.byte_start,
                byte_end=hypothesis.byte_end,
                confidence=hypothesis.confidence * 0.8,  # Reduce confidence
                data_type=alt_data_type,
                source=f"{hypothesis.source}_refined_datatype",
                reasoning=f"Refined data type from {hypothesis.data_type} to {alt_data_type}"
            )
            refined_hypotheses.append(refined_hyp)
        
        # Strategy 2: Try byte alignment adjustments
        for alignment in self.byte_alignments:
            aligned_start = (hypothesis.byte_start // alignment) * alignment
            if aligned_start != hypothesis.byte_start:
                byte_size = hypothesis.byte_end - hypothesis.byte_start + 1
                refined_hyp = AttributeHypothesis(
                    attribute_name=hypothesis.attribute_name,
                    byte_start=aligned_start,
                    byte_end=aligned_start + byte_size - 1,
                    confidence=hypothesis.confidence * 0.7,
                    data_type=hypothesis.data_type,
                    source=f"{hypothesis.source}_refined_alignment",
                    reasoning=f"Aligned to {alignment}-byte boundary"
                )
                refined_hypotheses.append(refined_hyp)
        
        # Strategy 3: Try nearby locations
        for offset in self.exploration_offsets[:3]:  # Limit to first 3 offsets
            new_start = hypothesis.byte_start + offset
            if new_start > 0:  # Ensure positive byte location
                byte_size = hypothesis.byte_end - hypothesis.byte_start + 1
                refined_hyp = AttributeHypothesis(
                    attribute_name=hypothesis.attribute_name,
                    byte_start=new_start,
                    byte_end=new_start + byte_size - 1,
                    confidence=hypothesis.confidence * 0.6,
                    data_type=hypothesis.data_type,
                    source=f"{hypothesis.source}_refined_location",
                    reasoning=f"Explored nearby location (offset: {offset})"
                )
                refined_hypotheses.append(refined_hyp)
        
        # Strategy 4: Try standard locations if available
        attr_name_lower = hypothesis.attribute_name.lower().replace(' ', '_')
        if attr_name_lower in self.standard_locations:
            start, end = self.standard_locations[attr_name_lower]
            refined_hyp = AttributeHypothesis(
                attribute_name=hypothesis.attribute_name,
                byte_start=start,
                byte_end=end,
                confidence=0.6,  # Medium confidence for standard locations
                data_type=hypothesis.data_type,
                source=f"{hypothesis.source}_refined_standard",
                reasoning=f"Using standard SEGY location for {hypothesis.attribute_name}"
            )
            refined_hypotheses.append(refined_hyp)
        
        # Sort by confidence and return top candidates
        refined_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return refined_hypotheses[:5]  # Return top 5 candidates