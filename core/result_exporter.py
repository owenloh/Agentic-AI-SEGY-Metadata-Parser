"""
ResultExporter - Multi-format exporter for SEGY parsing results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from models.data_models import (
    ParsedResults, AttributeHypothesis, GeometricInfo, 
    ValidationResult, ReasoningResult, ValidationStatus
)


class ResultExporter:
    """Multi-format exporter for comprehensive SEGY parsing results"""
    
    def __init__(self, output_base_path: Path):
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True)
    
    def export_results(self, results: ParsedResults) -> Dict[str, Path]:
        """
        Export results to all supported formats
        
        Args:
            results: Complete parsing results
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        base_filename = Path(results.filename).stem
        exported_files = {}
        
        # Export to JSON format
        json_path = self.export_json(results, base_filename)
        if json_path:
            exported_files['json'] = json_path
        
        # Export to TXT format
        txt_path = self.export_txt(results, base_filename)
        if txt_path:
            exported_files['txt'] = txt_path
        
        # Export to CSV format
        csv_path = self.export_csv(results, base_filename)
        if csv_path:
            exported_files['csv'] = csv_path
        
        return exported_files
    
    def export_json(self, results: ParsedResults, base_filename: str) -> Path:
        """Export results to machine-readable JSON format"""
        try:
            output_path = self.output_base_path / f"{base_filename}_parsing_results.json"
            

            
            # Convert results to JSON-serializable format
            json_data = {
                "metadata": {
                    "filename": results.filename,
                    "processing_time": results.processing_time,
                    "export_timestamp": datetime.now().isoformat(),
                    "confidence_summary": results.confidence_summary
                },
                "revision_info": {
                    "revision": results.revision_info.revision,
                    "confidence": results.revision_info.confidence.value if hasattr(results.revision_info.confidence, 'value') else results.revision_info.confidence,
                    "source": results.revision_info.source,
                    "reasoning": results.revision_info.reasoning
                },
                "attributes": [
                    self._attribute_to_dict(attr) for attr in results.attributes
                ],
                "geometric_info": self._geometric_info_to_dict(results.geometric_info),
                "validation_results": [
                    self._validation_result_to_dict(val) for val in results.validation_results
                ],
                "reasoning_chains": [
                    self._reasoning_result_to_dict(reason) for reason in results.reasoning_chains
                ],
                "fallback_strategies_used": results.fallback_strategies_used
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            return None
    
    def export_txt(self, results: ParsedResults, base_filename: str) -> Path:
        """Export results to human-readable TXT format"""
        try:
            output_path = self.output_base_path / f"{base_filename}_parsing_results.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write("AI-POWERED SEGY METADATA PARSER - PARSING RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                # File Information
                f.write("FILE INFORMATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Filename: {results.filename}\n")
                f.write(f"Processing Time: {results.processing_time:.2f} seconds\n")
                f.write(f"Export Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Revision Detection
                f.write("SEGY REVISION DETECTION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Detected Revision: {results.revision_info.revision}\n")
                f.write(f"Confidence Level: {results.revision_info.confidence.value if hasattr(results.revision_info.confidence, 'value') else results.revision_info.confidence}\n")
                f.write(f"Detection Source: {results.revision_info.source}\n")
                if results.revision_info.reasoning:
                    f.write(f"Reasoning: {results.revision_info.reasoning}\n")
                f.write("\n")
                
                # Confidence Summary
                f.write("CONFIDENCE SUMMARY\n")
                f.write("-" * 40 + "\n")
                for level, count in results.confidence_summary.items():
                    f.write(f"{level.capitalize()} Confidence: {count} attributes\n")
                f.write("\n")
                
                # Attribute-Byte Mappings
                f.write("ATTRIBUTE-BYTE MAPPINGS\n")
                f.write("-" * 40 + "\n")
                for attr in sorted(results.attributes, key=lambda x: x.confidence, reverse=True):
                    f.write(f"Attribute: {attr.attribute_name}\n")
                    f.write(f"  Byte Range: {attr.byte_start}-{attr.byte_end}\n")
                    f.write(f"  Data Type: {attr.data_type}\n")
                    f.write(f"  Confidence: {attr.confidence:.3f}\n")
                    f.write(f"  Source: {attr.source}\n")
                    validation_status = attr.validation_status
                    if hasattr(validation_status, 'value'):
                        validation_str = validation_status.value
                    elif validation_status:
                        validation_str = str(validation_status)
                    else:
                        validation_str = 'not_validated'
                    f.write(f"  Validation: {validation_str}\n")
                    if attr.reasoning:
                        f.write(f"  Reasoning: {attr.reasoning}\n")
                    f.write("\n")
                
                # Geometric Information
                f.write("GEOMETRIC INFORMATION\n")
                f.write("-" * 40 + "\n")
                
                # World Coordinates
                if results.geometric_info.world_coordinates:
                    f.write("World Coordinates:\n")
                    for coord_type, attr in results.geometric_info.world_coordinates.items():
                        f.write(f"  {coord_type}: bytes {attr.byte_start}-{attr.byte_end} "
                               f"(confidence: {attr.confidence:.3f})\n")
                    f.write("\n")
                
                # Azimuthal Angle
                if results.geometric_info.azimuthal_angle:
                    attr = results.geometric_info.azimuthal_angle
                    f.write(f"Azimuthal Angle: bytes {attr.byte_start}-{attr.byte_end} "
                           f"(confidence: {attr.confidence:.3f})\n\n")
                
                # Inline/Crossline
                if results.geometric_info.inline_crossline:
                    f.write("Inline/Crossline Mapping:\n")
                    for line_type, attr in results.geometric_info.inline_crossline.items():
                        f.write(f"  {line_type}: bytes {attr.byte_start}-{attr.byte_end} "
                               f"(confidence: {attr.confidence:.3f})\n")
                    f.write("\n")
                
                # Coordinate Reference System
                if results.geometric_info.coordinate_reference_system:
                    f.write(f"Coordinate Reference System: {results.geometric_info.coordinate_reference_system}\n")
                
                if results.geometric_info.projection_info:
                    f.write(f"Projection Info: {results.geometric_info.projection_info}\n")
                
                f.write("\n")
                
                # Validation Results
                if results.validation_results:
                    f.write("VALIDATION RESULTS\n")
                    f.write("-" * 40 + "\n")
                    for i, val_result in enumerate(results.validation_results, 1):
                        f.write(f"Validation {i}:\n")
                        f.write(f"  Status: {'PASSED' if val_result.passed else 'FAILED'}\n")
                        f.write(f"  Confidence: {val_result.confidence:.3f}\n")
                        if val_result.llm_evaluation:
                            f.write(f"  LLM Evaluation: {val_result.llm_evaluation}\n")
                        if val_result.issues:
                            f.write(f"  Issues: {', '.join(val_result.issues)}\n")
                        f.write("\n")
                
                # Chain-of-Thought Reasoning
                if results.reasoning_chains:
                    f.write("CHAIN-OF-THOUGHT REASONING\n")
                    f.write("-" * 40 + "\n")
                    for i, reasoning in enumerate(results.reasoning_chains, 1):
                        f.write(f"Reasoning Chain {i}:\n")
                        f.write(f"  Conclusion: {reasoning.conclusion}\n")
                        f.write(f"  Confidence: {reasoning.confidence:.3f}\n")
                        if reasoning.reasoning_steps:
                            f.write("  Steps:\n")
                            for j, step in enumerate(reasoning.reasoning_steps, 1):
                                f.write(f"    {j}. {step}\n")
                        if reasoning.alternative_interpretations:
                            f.write("  Alternatives:\n")
                            for alt in reasoning.alternative_interpretations:
                                f.write(f"    - {alt}\n")
                        f.write("\n")
                
                # Fallback Strategies
                if results.fallback_strategies_used:
                    f.write("FALLBACK STRATEGIES USED\n")
                    f.write("-" * 40 + "\n")
                    for strategy in results.fallback_strategies_used:
                        f.write(f"- {strategy}\n")
                    f.write("\n")
                
                # Footer
                f.write("=" * 80 + "\n")
                f.write("End of Report\n")
                f.write("=" * 80 + "\n")
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting TXT: {e}")
            return None
    
    def export_csv(self, results: ParsedResults, base_filename: str) -> Path:
        """Export results to CSV format for quick overview"""
        try:
            output_path = self.output_base_path / f"{base_filename}_parsing_summary.csv"
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Attribute Name', 'Byte Start', 'Byte End', 'Data Type',
                    'Confidence', 'Source', 'Validation Status', 'Reasoning'
                ])
                
                # Attribute data
                for attr in sorted(results.attributes, key=lambda x: x.confidence, reverse=True):
                    writer.writerow([
                        attr.attribute_name,
                        attr.byte_start,
                        attr.byte_end,
                        attr.data_type,
                        f"{attr.confidence:.3f}",
                        attr.source,
                        attr.validation_status.value if hasattr(attr.validation_status, 'value') else (str(attr.validation_status) if attr.validation_status else 'not_validated'),
                        attr.reasoning[:100] + '...' if len(attr.reasoning) > 100 else attr.reasoning
                    ])
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return None
    
    def _attribute_to_dict(self, attr: AttributeHypothesis) -> Dict[str, Any]:
        """Convert AttributeHypothesis to dictionary"""
        return {
            "attribute_name": attr.attribute_name,
            "byte_start": attr.byte_start,
            "byte_end": attr.byte_end,
            "confidence": attr.confidence,
            "data_type": attr.data_type,
            "source": attr.source,
            "reasoning": attr.reasoning,
            "validation_status": attr.validation_status.value if hasattr(attr.validation_status, 'value') else (str(attr.validation_status) if attr.validation_status else None)
        }
    
    def _geometric_info_to_dict(self, geo_info: GeometricInfo) -> Dict[str, Any]:
        """Convert GeometricInfo to dictionary"""
        return {
            "world_coordinates": {
                coord_type: self._attribute_to_dict(attr)
                for coord_type, attr in geo_info.world_coordinates.items()
            },
            "azimuthal_angle": (
                self._attribute_to_dict(geo_info.azimuthal_angle)
                if geo_info.azimuthal_angle else None
            ),
            "inline_crossline": {
                line_type: self._attribute_to_dict(attr)
                for line_type, attr in geo_info.inline_crossline.items()
            },
            "coordinate_reference_system": geo_info.coordinate_reference_system,
            "projection_info": geo_info.projection_info
        }
    
    def _validation_result_to_dict(self, val_result: ValidationResult) -> Dict[str, Any]:
        """Convert ValidationResult to dictionary"""
        return {
            "passed": val_result.passed,
            "confidence": val_result.confidence,
            "llm_evaluation": val_result.llm_evaluation,
            "issues": val_result.issues,
            "statistical_profile": {
                "mean": val_result.statistical_profile.mean,
                "median": val_result.statistical_profile.median,
                "std": val_result.statistical_profile.std,
                "min_val": val_result.statistical_profile.min_val,
                "max_val": val_result.statistical_profile.max_val,
                "detected_type": val_result.statistical_profile.detected_type
            } if val_result.statistical_profile else None,
            "suggestions": [
                {
                    "suggestion_type": sug.suggestion_type,
                    "parameters": sug.parameters,
                    "reasoning": sug.reasoning,
                    "confidence": sug.confidence
                }
                for sug in val_result.suggestions
            ]
        }
    
    def _reasoning_result_to_dict(self, reasoning: ReasoningResult) -> Dict[str, Any]:
        """Convert ReasoningResult to dictionary"""
        return {
            "conclusion": reasoning.conclusion,
            "reasoning_steps": reasoning.reasoning_steps,
            "confidence": reasoning.confidence,
            "alternative_interpretations": reasoning.alternative_interpretations,
            "requires_validation": reasoning.requires_validation
        }