"""
AttributeOntology system for managing SEGY attribute schemas and validation rules.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from models.data_models import AttributeSchema, ValidationRule


class AttributeOntology:
    """Manages SEGY attribute schemas and provides validation guidance"""
    
    def __init__(self, ontology_path: Optional[Path] = None):
        self.ontology_path = ontology_path or Path("segy_attribute_ontology.json")
        self.schemas: Dict[str, AttributeSchema] = {}
        self._load_ontology()
    
    def _load_ontology(self) -> None:
        """Load attribute ontology from JSON file or create default"""
        if self.ontology_path.exists():
            try:
                with open(self.ontology_path, 'r') as f:
                    data = json.load(f)
                self._parse_ontology_data(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading ontology: {e}. Creating default ontology.")
                self._create_default_ontology()
        else:
            self._create_default_ontology()
    
    def _parse_ontology_data(self, data: Dict) -> None:
        """Parse loaded JSON data into AttributeSchema objects"""
        for attr_name, attr_data in data.get("attributes", {}).items():
            validation_rules = []
            for rule_data in attr_data.get("validation_rules", []):
                validation_rules.append(ValidationRule(
                    rule_type=rule_data["rule_type"],
                    parameters=rule_data["parameters"],
                    description=rule_data["description"]
                ))
            
            schema = AttributeSchema(
                name=attr_name,
                expected_types=attr_data["expected_types"],
                expected_range=tuple(attr_data["expected_range"]),
                standard_locations=attr_data["standard_locations"],
                validation_rules=validation_rules,
                cross_checks=attr_data.get("cross_checks", []),
                description=attr_data.get("description", "")
            )
            self.schemas[attr_name] = schema
    
    def _create_default_ontology(self) -> None:
        """Create default SEGY attribute ontology"""
        default_schemas = {
            "source_x": AttributeSchema(
                name="source_x",
                expected_types=["int32", "float32"],
                expected_range=(-1e8, 1e8),
                standard_locations={
                    "0": [(73, 76)],
                    "1.0": [(73, 76)],
                    "2.0": [(73, 76)],
                    "2.1": [(73, 76)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": -1e8, "max": 1e8}, "Valid coordinate range"),
                    ValidationRule("data_type", {"types": ["int32", "float32"]}, "Numeric coordinate")
                ],
                cross_checks=["source_y", "receiver_x"],
                description="X coordinate of seismic source"
            ),
            "source_y": AttributeSchema(
                name="source_y",
                expected_types=["int32", "float32"],
                expected_range=(-1e8, 1e8),
                standard_locations={
                    "0": [(77, 80)],
                    "1.0": [(77, 80)],
                    "2.0": [(77, 80)],
                    "2.1": [(77, 80)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": -1e8, "max": 1e8}, "Valid coordinate range"),
                    ValidationRule("data_type", {"types": ["int32", "float32"]}, "Numeric coordinate")
                ],
                cross_checks=["source_x", "receiver_y"],
                description="Y coordinate of seismic source"
            ),
            "receiver_x": AttributeSchema(
                name="receiver_x",
                expected_types=["int32", "float32"],
                expected_range=(-1e8, 1e8),
                standard_locations={
                    "0": [(81, 84)],
                    "1.0": [(81, 84)],
                    "2.0": [(81, 84)],
                    "2.1": [(81, 84)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": -1e8, "max": 1e8}, "Valid coordinate range"),
                    ValidationRule("data_type", {"types": ["int32", "float32"]}, "Numeric coordinate")
                ],
                cross_checks=["receiver_y", "source_x"],
                description="X coordinate of seismic receiver"
            ),
            "receiver_y": AttributeSchema(
                name="receiver_y",
                expected_types=["int32", "float32"],
                expected_range=(-1e8, 1e8),
                standard_locations={
                    "0": [(85, 88)],
                    "1.0": [(85, 88)],
                    "2.0": [(85, 88)],
                    "2.1": [(85, 88)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": -1e8, "max": 1e8}, "Valid coordinate range"),
                    ValidationRule("data_type", {"types": ["int32", "float32"]}, "Numeric coordinate")
                ],
                cross_checks=["receiver_x", "source_y"],
                description="Y coordinate of seismic receiver"
            ),
            "inline_number": AttributeSchema(
                name="inline_number",
                expected_types=["int32"],
                expected_range=(1, 1000000),
                standard_locations={
                    "0": [(189, 192)],
                    "1.0": [(189, 192)],
                    "2.0": [(189, 192)],
                    "2.1": [(189, 192)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": 1, "max": 1000000}, "Valid inline range"),
                    ValidationRule("data_type", {"types": ["int32"]}, "Integer inline number")
                ],
                cross_checks=["crossline_number"],
                description="Inline number for 3D surveys"
            ),
            "crossline_number": AttributeSchema(
                name="crossline_number",
                expected_types=["int32"],
                expected_range=(1, 1000000),
                standard_locations={
                    "0": [(193, 196)],
                    "1.0": [(193, 196)],
                    "2.0": [(193, 196)],
                    "2.1": [(193, 196)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": 1, "max": 1000000}, "Valid crossline range"),
                    ValidationRule("data_type", {"types": ["int32"]}, "Integer crossline number")
                ],
                cross_checks=["inline_number"],
                description="Crossline number for 3D surveys"
            ),
            "cdp_x": AttributeSchema(
                name="cdp_x",
                expected_types=["int32", "float32"],
                expected_range=(-1e8, 1e8),
                standard_locations={
                    "0": [(181, 184)],
                    "1.0": [(181, 184)],
                    "2.0": [(181, 184)],
                    "2.1": [(181, 184)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": -1e8, "max": 1e8}, "Valid coordinate range"),
                    ValidationRule("data_type", {"types": ["int32", "float32"]}, "Numeric coordinate")
                ],
                cross_checks=["cdp_y"],
                description="X coordinate of CDP (Common Depth Point)"
            ),
            "cdp_y": AttributeSchema(
                name="cdp_y",
                expected_types=["int32", "float32"],
                expected_range=(-1e8, 1e8),
                standard_locations={
                    "0": [(185, 188)],
                    "1.0": [(185, 188)],
                    "2.0": [(185, 188)],
                    "2.1": [(185, 188)]
                },
                validation_rules=[
                    ValidationRule("range", {"min": -1e8, "max": 1e8}, "Valid coordinate range"),
                    ValidationRule("data_type", {"types": ["int32", "float32"]}, "Numeric coordinate")
                ],
                cross_checks=["cdp_x"],
                description="Y coordinate of CDP (Common Depth Point)"
            )
        }
        
        self.schemas = default_schemas
        self._save_ontology()
    
    def _save_ontology(self) -> None:
        """Save current ontology to JSON file"""
        data = {"attributes": {}}
        for name, schema in self.schemas.items():
            data["attributes"][name] = {
                "expected_types": schema.expected_types,
                "expected_range": list(schema.expected_range),
                "standard_locations": schema.standard_locations,
                "validation_rules": [
                    {
                        "rule_type": rule.rule_type,
                        "parameters": rule.parameters,
                        "description": rule.description
                    }
                    for rule in schema.validation_rules
                ],
                "cross_checks": schema.cross_checks,
                "description": schema.description
            }
        
        with open(self.ontology_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_attribute_schema(self, attribute_name: str) -> Optional[AttributeSchema]:
        """Get schema for a specific attribute"""
        return self.schemas.get(attribute_name)
    
    def get_standard_locations(self, revision: str) -> Dict[str, List[Tuple[int, int]]]:
        """Get standard byte locations for all attributes for a given revision"""
        locations = {}
        for name, schema in self.schemas.items():
            if revision in schema.standard_locations:
                locations[name] = schema.standard_locations[revision]
        return locations
    
    def get_all_attributes(self) -> List[str]:
        """Get list of all known attribute names"""
        return list(self.schemas.keys())
    
    def get_attributes_by_revision(self, revision: str) -> Dict[str, AttributeSchema]:
        """Get attributes relevant to a specific SEGY revision."""
        # For now, return all attributes - in a full implementation, 
        # this would filter based on revision-specific availability
        return self.schemas.copy()
    
    def validate_against_schema(self, attribute_name: str, data_type: str, 
                              value_range: Tuple[float, float]) -> bool:
        """Basic schema validation for attribute hypothesis"""
        schema = self.get_attribute_schema(attribute_name)
        if not schema:
            return False
        
        # Check data type
        if data_type not in schema.expected_types:
            return False
        
        # Check range overlap
        schema_min, schema_max = schema.expected_range
        value_min, value_max = value_range
        
        # Check if ranges overlap
        return not (value_max < schema_min or value_min > schema_max)