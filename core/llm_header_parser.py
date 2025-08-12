"""
LLM-based header parser for SEGY files.

This module provides intelligent parsing of SEGY textual headers using LLM providers
with ontology-guided hypothesis generation and fallback strategies.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from core.llm_provider import LLMFactory, LLMResponse, ChainOfThoughtReasoner
from core.attribute_ontology import AttributeOntology, AttributeSchema
from core.segy_file_handler import SEGYFileHandler


@dataclass
class AttributeHypothesis:
    """A hypothesis about an attribute's location and properties."""
    attribute_name: str
    byte_start: int
    byte_end: int
    data_type: str
    confidence: float
    source: str  # 'llm', 'ontology', 'fallback', 'binary_analysis'
    reasoning: Optional[str] = None
    validation_status: str = 'pending'  # 'pending', 'validated', 'failed', 'refined'


@dataclass
class RevisionInfo:
    """Information about SEGY revision/format."""
    revision: str  # '0', '1', '2', '2.1', 'None'
    confidence: float
    source: str
    reasoning: Optional[str] = None


@dataclass
class GeometricData:
    """Geometric information extracted from textual header."""
    corner_points: List[Dict[str, float]]
    grid_origin: Optional[Dict[str, float]]
    coordinate_system: Optional[Dict[str, str]]
    survey_geometry: Optional[Dict[str, float]]
    survey_info: Optional[Dict[str, str]]
    processing_info: List[str]
    additional_geometry: Dict[str, Any]


@dataclass
class HeaderParseResult:
    """Result of header parsing operation."""
    textual_header: str
    revision_info: RevisionInfo
    attribute_hypotheses: List[AttributeHypothesis]
    geometric_data: Optional[GeometricData]
    parsing_success: bool
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class LLMHeaderParser:
    """
    Intelligent SEGY header parser using LLM providers with ontology guidance.
    """
    
    def __init__(self, llm_factory: LLMFactory, ontology: AttributeOntology):
        self.llm_factory = llm_factory
        self.ontology = ontology
        self.reasoner = ChainOfThoughtReasoner(llm_factory)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Enhanced parsing prompts for comprehensive SEGY header analysis
        self.ATTRIBUTE_EXTRACTION_PROMPT = """
You are a senior geophysicist and SEGY format expert.

Given the SEGY textual header below, identify EVERY metadata field whose byte locations 
are explicitly stated or can be clearly inferred from the header content.

Respond with a JSON list of dictionaries in this exact schema:
[{"attribute": "Exact field name from header", "bytelocation": "startbyte-endbyte", "confidence": "high|medium|low", "reasoning": "Brief explanation"}, ...]

Rules:
- confidence 'high': header explicitly gives the byte range (e.g., "bytes 73-76")
- confidence 'medium': clear inference from context (e.g., "byte 133, 4-byte integer" implies 133-136)
- confidence 'low': ambiguous or approximate guess
- If only startbyte given with size info, calculate endbyte (startbyte + size - 1)
- If only startbyte given without size, use "startbyte-None"
- Include ALL attributes found, even if repeated with different contexts
- Look for ANY mention of byte locations, positions, or trace header definitions
- Check for standard SEGY fields even if not explicitly mentioned (source_x at 73-76, etc.)
- Ignore line labels (C01, C02, etc.) - focus on the actual content
- Normalize byte ranges to "startbyte-endbyte" format
- Be comprehensive - don't miss any potential attribute mappings
- No prose - only raw JSON

SEGY Textual Header:
"""

        self.GEOMETRIC_EXTRACTION_PROMPT = """
You are a senior geophysicist and SEGY format expert specializing in survey geometry.

Given the SEGY textual header below, extract ALL geometric and survey information present.

Respond with a JSON dictionary in this exact schema:
{
  "corner_points": [{"line": number, "cdp": number, "easting": number, "northing": number}, ...],
  "grid_origin": {"easting": number, "northing": number, "line_direction": number},
  "coordinate_system": {"datum": "string", "projection": "string", "zone": "string"},
  "survey_geometry": {
    "cdp_spacing": number,
    "line_spacing": number,
    "cdp_origin": number,
    "line_origin": number,
    "cdp_increment": number,
    "line_increment": number,
    "cdp_direction": number,
    "line_direction": number
  },
  "survey_info": {
    "survey_name": "string",
    "line_name": "string",
    "data_type": "string",
    "polarity": "string"
  },
  "processing_info": ["list of processing steps"],
  "additional_geometry": {"any_other_geometric_info": "value"}
}

Rules:
- Extract ALL numeric values with their proper units
- Include corner points, grid origins, coordinate systems
- Capture survey geometry parameters (spacing, origins, directions)
- Extract processing sequence information
- Include survey metadata (names, types, polarity)
- If information is missing, use null
- Convert all numeric strings to actual numbers
- Preserve units in separate fields where applicable
- Be comprehensive - capture everything geometric/survey related
- No prose - only raw JSON

SEGY Textual Header:
"""

        self.REVISION_DETECTION_PROMPT = """
You are a SEGY format expert.

From the textual header below, detect the SEGY revision/version.

Respond with exactly one JSON dict:
{"revision": "0|1|2|2.1|None", "confidence": "high|medium|low", "reasoning": "Brief explanation"}

Rules:
- 'high': exact phrase like 'SEG-Y revision 1.0' or 'SEGY Rev 2' appears
- 'medium': strong contextual clues about format version
- 'low': weak or conflicting hints
- If no evidence, use {"revision": "None", "confidence": "high", "reasoning": "No revision information found"}
- Ignore line labels (C01, C02, etc.)
- Only the JSON dict - no extra text

SEGY Textual Header:
"""

    def parse_header(self, segy_file_path: Path) -> HeaderParseResult:
        """
        Parse SEGY header and generate attribute hypotheses.
        
        Args:
            segy_file_path: Path to SEGY file
            
        Returns:
            HeaderParseResult with all parsing information
        """
        import time
        start_time = time.time()
        
        try:
            # Extract textual header
            segy_handler = SEGYFileHandler(segy_file_path)
            if not segy_handler.open_file():
                raise Exception("Could not open SEGY file")
            
            textual_headers = segy_handler.get_textual_headers()
            textual_header = '\n'.join(textual_headers) if textual_headers else ""
            segy_handler.close_file()
            
            if not textual_header:
                return HeaderParseResult(
                    textual_header="",
                    revision_info=RevisionInfo("None", 0.0, "error"),
                    attribute_hypotheses=[],
                    geometric_data=None,
                    parsing_success=False,
                    error_message="Could not extract textual header"
                )
            
            # Parse revision information
            revision_info = self._detect_revision(textual_header)
            
            # Generate attribute hypotheses
            attribute_hypotheses = self._generate_attribute_hypotheses(textual_header, revision_info)
            
            # Extract geometric information
            geometric_data = self._extract_geometric_information(textual_header)
            
            processing_time = time.time() - start_time
            
            return HeaderParseResult(
                textual_header=textual_header,
                revision_info=revision_info,
                attribute_hypotheses=attribute_hypotheses,
                geometric_data=geometric_data,
                parsing_success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Header parsing failed: {e}")
            processing_time = time.time() - start_time
            
            return HeaderParseResult(
                textual_header="",
                revision_info=RevisionInfo("None", 0.0, "error"),
                attribute_hypotheses=[],
                geometric_data=None,
                parsing_success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _detect_revision(self, textual_header: str) -> RevisionInfo:
        """Detect SEGY revision from textual header."""
        provider = self.llm_factory.get_available_provider()
        
        if not provider:
            # Fallback to simple pattern matching
            return self._fallback_revision_detection(textual_header)
        
        try:
            prompt = self.REVISION_DETECTION_PROMPT + textual_header
            response = provider.invoke_prompt(
                "You are a SEGY format expert detecting revision information.",
                prompt,
                max_tokens=200
            )
            
            if response.success:
                revision_data = self._parse_revision_response(response.content)
                if revision_data:
                    return RevisionInfo(
                        revision=revision_data.get('revision', 'None'),
                        confidence=self._normalize_confidence(revision_data.get('confidence', 'low')),
                        source='llm',
                        reasoning=revision_data.get('reasoning', '')
                    )
            
        except Exception as e:
            self.logger.warning(f"LLM revision detection failed: {e}")
        
        # Fallback to pattern matching
        return self._fallback_revision_detection(textual_header)
    
    def _generate_attribute_hypotheses(self, textual_header: str, revision_info: RevisionInfo) -> List[AttributeHypothesis]:
        """Generate attribute hypotheses using LLM and ontology guidance."""
        hypotheses = []
        
        # 1. LLM-based extraction
        llm_hypotheses = self._llm_attribute_extraction(textual_header)
        hypotheses.extend(llm_hypotheses)
        
        # 2. Ontology-guided enhancement
        ontology_hypotheses = self._ontology_guided_hypotheses(textual_header, revision_info)
        hypotheses.extend(ontology_hypotheses)
        
        # 3. Remove duplicates and merge similar hypotheses
        hypotheses = self._merge_duplicate_hypotheses(hypotheses)
        
        # 4. Apply confidence scoring based on multiple sources
        hypotheses = self._apply_confidence_scoring(hypotheses)
        
        return hypotheses
    
    def _llm_attribute_extraction(self, textual_header: str) -> List[AttributeHypothesis]:
        """Extract attributes using LLM."""
        provider = self.llm_factory.get_available_provider()
        
        if not provider:
            self.logger.warning("No LLM provider available for attribute extraction")
            return []
        
        try:
            prompt = self.ATTRIBUTE_EXTRACTION_PROMPT + textual_header
            response = provider.invoke_prompt(
                "You are extracting SEGY attribute byte locations from textual headers.",
                prompt,
                max_tokens=4000
            )
            
            if response.success:
                return self._parse_attribute_response(response.content, 'llm')
            
        except Exception as e:
            self.logger.warning(f"LLM attribute extraction failed: {e}")
        
        return []
    

    
    def _parse_attribute_response(self, response_content: str, source: str) -> List[AttributeHypothesis]:
        """Parse LLM response into AttributeHypothesis objects."""
        hypotheses = []
        
        try:
            # Clean up response
            content = response_content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON
            import ast
            data = ast.literal_eval(content)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and all(key in item for key in ['attribute', 'bytelocation', 'confidence']):
                        # Parse byte location
                        byte_start, byte_end = self._parse_byte_location(item['bytelocation'])
                        
                        hypothesis = AttributeHypothesis(
                            attribute_name=item['attribute'],
                            byte_start=byte_start,
                            byte_end=byte_end,
                            data_type=self._infer_data_type(item.get('reasoning', '')),
                            confidence=self._normalize_confidence(item['confidence']),
                            source=source,
                            reasoning=item.get('reasoning', '')
                        )
                        hypotheses.append(hypothesis)
            
        except Exception as e:
            self.logger.error(f"Failed to parse attribute response: {e}")
        
        return hypotheses
    
    def _parse_byte_location(self, byte_location: str) -> Tuple[int, int]:
        """Parse byte location string into start and end bytes."""
        # Handle various formats: "73-76", "73:76", "73 to 76", "73-None", etc.
        import re
        
        # Extract numbers
        numbers = re.findall(r'\d+', byte_location)
        
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        elif len(numbers) == 1:
            start = int(numbers[0])
            # If "None" is mentioned, return start with estimated end
            if 'none' in byte_location.lower():
                return start, start + 3  # Assume 4-byte default
            else:
                return start, start + 3  # Assume 4-byte default
        else:
            return 0, 3  # Default fallback
    
    def _infer_data_type(self, reasoning: str) -> str:
        """Infer data type from reasoning text."""
        reasoning_lower = reasoning.lower()
        
        if 'float' in reasoning_lower or 'decimal' in reasoning_lower or 'double' in reasoning_lower:
            if '64' in reasoning_lower or 'double' in reasoning_lower:
                return 'float64'
            else:
                return 'float32'
        elif 'int' in reasoning_lower or 'integer' in reasoning_lower:
            # Check for specific byte size mentions first
            if '2-byte' in reasoning_lower or '16' in reasoning_lower or 'short' in reasoning_lower:
                return 'int16'
            elif '8-byte' in reasoning_lower or '64' in reasoning_lower or 'long' in reasoning_lower:
                return 'int64'
            else:
                return 'int32'
        elif 'byte' in reasoning_lower:
            if '2' in reasoning_lower:
                return 'int16'
            elif '4' in reasoning_lower:
                return 'int32'
            elif '8' in reasoning_lower:
                return 'int64'
            else:
                return 'int32'
        else:
            return 'int32'  # Default
    
    def _normalize_confidence(self, confidence: str) -> float:
        """Normalize confidence string to float."""
        if isinstance(confidence, (int, float)):
            return float(confidence)
        
        confidence_lower = str(confidence).lower().strip()
        
        confidence_map = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.4,
            'very high': 0.95,
            'very low': 0.2,
            'uncertain': 0.3
        }
        
        return confidence_map.get(confidence_lower, 0.5)
    
    def _parse_revision_response(self, response_content: str) -> Optional[Dict[str, str]]:
        """Parse revision detection response."""
        try:
            content = response_content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            import ast
            data = ast.literal_eval(content)
            
            if isinstance(data, dict) and 'revision' in data and 'confidence' in data:
                return data
                
        except Exception as e:
            self.logger.error(f"Failed to parse revision response: {e}")
        
        return None
    
    def _fallback_revision_detection(self, textual_header: str) -> RevisionInfo:
        """Fallback revision detection using pattern matching."""
        header_lower = textual_header.lower()
        
        # Look for explicit revision mentions
        revision_patterns = [
            (r'seg-?y\s+rev(?:ision)?\s*2\.1', '2.1'),
            (r'seg-?y\s+rev(?:ision)?\s*2\.0', '2.0'),
            (r'seg-?y\s+rev(?:ision)?\s*2', '2'),
            (r'seg-?y\s+rev(?:ision)?\s*1\.0', '1.0'),
            (r'seg-?y\s+rev(?:ision)?\s*1', '1'),
            (r'seg-?y\s+rev(?:ision)?\s*0', '0'),
        ]
        
        for pattern, revision in revision_patterns:
            if re.search(pattern, header_lower):
                return RevisionInfo(
                    revision=revision,
                    confidence=0.8,
                    source='pattern_matching',
                    reasoning=f"Found pattern: {pattern}"
                )
        
        # Default to None if no revision found
        return RevisionInfo(
            revision='None',
            confidence=0.9,
            source='pattern_matching',
            reasoning='No revision information found in header'
        )
    
    def _merge_duplicate_hypotheses(self, hypotheses: List[AttributeHypothesis]) -> List[AttributeHypothesis]:
        """Merge duplicate or very similar hypotheses."""
        if not hypotheses:
            return []
        
        merged = []
        used_indices = set()
        
        for i, hyp1 in enumerate(hypotheses):
            if i in used_indices:
                continue
            
            # Find similar hypotheses
            similar = [hyp1]
            used_indices.add(i)
            
            for j, hyp2 in enumerate(hypotheses[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._are_hypotheses_similar(hyp1, hyp2):
                    similar.append(hyp2)
                    used_indices.add(j)
            
            # Merge similar hypotheses
            if len(similar) > 1:
                merged_hyp = self._merge_similar_hypotheses(similar)
                merged.append(merged_hyp)
            else:
                merged.append(hyp1)
        
        return merged
    
    def _are_hypotheses_similar(self, hyp1: AttributeHypothesis, hyp2: AttributeHypothesis) -> bool:
        """Check if two hypotheses are similar enough to merge."""
        # Same attribute name (case insensitive)
        if hyp1.attribute_name.lower() != hyp2.attribute_name.lower():
            return False
        
        # Overlapping byte ranges
        range1 = set(range(hyp1.byte_start, hyp1.byte_end + 1))
        range2 = set(range(hyp2.byte_start, hyp2.byte_end + 1))
        
        overlap = len(range1.intersection(range2))
        min_range_size = min(len(range1), len(range2))
        
        # Consider similar if >50% overlap
        return overlap / min_range_size > 0.5
    
    def _merge_similar_hypotheses(self, hypotheses: List[AttributeHypothesis]) -> AttributeHypothesis:
        """Merge a list of similar hypotheses into one."""
        # Use the hypothesis with highest confidence as base
        base = max(hypotheses, key=lambda h: h.confidence)
        
        # Combine sources and reasoning
        sources = list(set(h.source for h in hypotheses))
        reasonings = [h.reasoning for h in hypotheses if h.reasoning]
        
        # Average confidence weighted by source reliability
        source_weights = {'llm': 1.0, 'ontology': 0.8, 'fallback': 0.6, 'binary_analysis': 0.7}
        total_weight = sum(source_weights.get(h.source, 0.5) * h.confidence for h in hypotheses)
        total_weights = sum(source_weights.get(h.source, 0.5) for h in hypotheses)
        
        merged_confidence = total_weight / total_weights if total_weights > 0 else base.confidence
        
        return AttributeHypothesis(
            attribute_name=base.attribute_name,
            byte_start=base.byte_start,
            byte_end=base.byte_end,
            data_type=base.data_type,
            confidence=min(merged_confidence, 1.0),
            source='+'.join(sources),
            reasoning='; '.join(reasonings) if reasonings else base.reasoning
        )
    
    def _apply_confidence_scoring(self, hypotheses: List[AttributeHypothesis]) -> List[AttributeHypothesis]:
        """Apply final confidence scoring based on multiple factors."""
        for hypothesis in hypotheses:
            # Adjust confidence based on source reliability
            if hypothesis.source == 'llm':
                # LLM confidence is already set
                pass
            elif hypothesis.source == 'ontology':
                # Ontology-based hypotheses get medium confidence
                hypothesis.confidence = min(hypothesis.confidence, 0.7)
            elif 'fallback' in hypothesis.source:
                # Fallback strategies get lower confidence
                hypothesis.confidence = min(hypothesis.confidence, 0.5)
            
            # Boost confidence if multiple sources agree
            if '+' in hypothesis.source:
                hypothesis.confidence = min(hypothesis.confidence * 1.2, 1.0)
            
            # Penalize if byte range seems unusual
            byte_range_size = hypothesis.byte_end - hypothesis.byte_start + 1
            if byte_range_size > 8 or byte_range_size < 1:
                hypothesis.confidence *= 0.8
        
        return hypotheses    

    def _extract_geometric_information(self, textual_header: str) -> Optional[GeometricData]:
        """Extract comprehensive geometric information from textual header."""
        provider = self.llm_factory.get_available_provider()
        
        if not provider:
            self.logger.warning("No LLM provider available for geometric extraction")
            return None
        
        try:
            prompt = self.GEOMETRIC_EXTRACTION_PROMPT + textual_header
            response = provider.invoke_prompt(
                "You are extracting comprehensive geometric information from SEGY textual headers.",
                prompt,
                max_tokens=4000
            )
            
            if response.success:
                return self._parse_geometric_response(response.content)
            else:
                self.logger.warning(f"Geometric extraction failed: {response.error_message}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Geometric extraction error: {e}")
            return None
    
    def _parse_geometric_response(self, response_content: str) -> Optional[GeometricData]:
        """Parse LLM response into GeometricData object."""
        try:
            # Clean up response
            content = response_content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            # Parse JSON
            import json
            data = json.loads(content)
            
            return GeometricData(
                corner_points=data.get('corner_points', []),
                grid_origin=data.get('grid_origin'),
                coordinate_system=data.get('coordinate_system'),
                survey_geometry=data.get('survey_geometry'),
                survey_info=data.get('survey_info'),
                processing_info=data.get('processing_info', []),
                additional_geometry=data.get('additional_geometry', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse geometric response: {e}")
            return None
    

    
    def _generate_attribute_hypotheses(self, textual_header: str, revision_info: RevisionInfo) -> List[AttributeHypothesis]:
        """Generate attribute hypotheses purely from textual header with ontology guidance."""
        
        # Only extract attributes that have explicit or inferable byte location information
        # in the textual header. Use ontology to guide what to look for, but don't create
        # hypotheses without textual evidence.
        
        hypotheses = self._ontology_guided_textual_extraction(textual_header, revision_info)
        
        # Remove duplicates and apply confidence scoring
        hypotheses = self._merge_duplicate_hypotheses(hypotheses)
        hypotheses = self._apply_confidence_scoring(hypotheses)
        
        return hypotheses 
   
    def _ontology_guided_textual_extraction(self, textual_header: str, revision_info: RevisionInfo) -> List[AttributeHypothesis]:
        """
        Extract attributes using ontology guidance but only from textual header evidence.
        
        This method uses the ontology to know what attributes to look for, but only creates
        hypotheses if there's actual byte location information in the textual header.
        """
        provider = self.llm_factory.get_available_provider()
        
        if not provider:
            self.logger.warning("No LLM provider available for ontology-guided extraction")
            return []
        
        # Get relevant attributes from ontology to guide the search
        relevant_attributes = self.ontology.get_attributes_by_revision(revision_info.revision)
        if not relevant_attributes:
            # Fallback to common SEGY attributes
            relevant_attributes = self.ontology.get_attributes_by_revision('default')
        
        # Create a guided prompt that tells the LLM what to look for
        attribute_guidance = []
        for attr_name, attr_schema in relevant_attributes.items():
            attribute_guidance.append({
                'name': attr_name,
                'description': attr_schema.description,
                'expected_types': attr_schema.expected_types,
                'common_names': [attr_name.replace('_', ' '), attr_name.upper(), attr_name.title()]
            })
        
        guided_prompt = f"""
You are a senior geophysicist and SEGY format expert.

Given the SEGY textual header below, identify ONLY those metadata fields whose byte locations 
are explicitly stated or can be clearly inferred from the header content.

IMPORTANT: Do NOT create hypotheses for attributes unless you can find actual byte location 
information in the textual header. If an attribute is not mentioned with byte locations, 
ignore it completely.

Here are the types of attributes to look for (but only include if byte locations are found):
{self._format_attribute_guidance(attribute_guidance)}

Respond with a JSON list of dictionaries in this exact schema:
[{{"attribute": "Exact field name from header", "bytelocation": "startbyte-endbyte", "confidence": "high|medium|low", "reasoning": "Brief explanation with evidence from header"}}, ...]

Rules:
- confidence 'high': header explicitly gives the byte range (e.g., "bytes 73-76")
- confidence 'medium': clear inference from context (e.g., "byte 133, 4-byte integer" implies 133-136)
- confidence 'low': ambiguous but reasonable guess with some textual evidence
- ONLY include attributes where you found byte location information in the header
- If you cannot find byte location info for an attribute, DO NOT include it
- Include reasoning that quotes the specific text from the header
- No prose - only raw JSON

SEGY Textual Header:
{textual_header}
"""
        
        try:
            response = provider.invoke_prompt(
                "You are extracting SEGY attributes with byte locations from textual headers using ontology guidance.",
                guided_prompt,
                max_tokens=6000
            )
            
            if response.success:
                return self._parse_attribute_response(response.content, 'ontology_guided_llm')
            else:
                self.logger.warning(f"Ontology-guided extraction failed: {response.error_message}")
                return []
                
        except Exception as e:
            self.logger.warning(f"Ontology-guided extraction error: {e}")
            return []
    
    def _format_attribute_guidance(self, attribute_guidance: List[Dict]) -> str:
        """Format attribute guidance for the LLM prompt."""
        guidance_text = ""
        for attr in attribute_guidance[:20]:  # Limit to top 20 to avoid prompt bloat
            guidance_text += f"- {attr['name']}: {attr['description']}\n"
            guidance_text += f"  Common names: {', '.join(attr['common_names'])}\n"
            guidance_text += f"  Expected types: {', '.join(attr['expected_types'])}\n\n"
        
        return guidance_text