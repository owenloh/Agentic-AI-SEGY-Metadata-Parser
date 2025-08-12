#!/usr/bin/env python3
"""
Configuration System - Centralized configuration management for the AI-Powered SEGY Metadata Parser.

This module provides configuration presets and management for different use cases:
- Fast processing for quick analysis
- Accurate processing for detailed analysis
- Balanced processing for general use
- Custom configurations for specific needs
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict

from enhanced_segy_parser import ParsingConfig
from core.performance_optimizer import PerformanceOptimizer, ConfigurationTuner


@dataclass
class SystemConfiguration:
    """System-wide configuration settings."""
    cache_enabled: bool = True
    cache_max_age_days: int = 30
    default_output_directory: str = "output"
    log_level: str = "INFO"
    max_concurrent_files: int = 1
    auto_cleanup_temp_files: bool = True


class ConfigurationManager:
    """Centralized configuration management for the AI-Powered SEGY Metadata Parser."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(".config")
        self.config_dir.mkdir(exist_ok=True)
        
        self.optimizer = PerformanceOptimizer()
        self.tuner = ConfigurationTuner(self.optimizer)
        
        # Load system configuration
        self.system_config = self._load_system_config()
        
        # Predefined configurations
        self._init_predefined_configs()
    
    def _init_predefined_configs(self):
        """Initialize predefined configuration presets."""
        self.presets = {
            'fast': ParsingConfig(
                max_validation_attempts=1,
                sample_size_for_validation=25,
                enable_chain_of_thought=False,
                enable_fallback_strategies=False,
                enable_hypothesis_refinement=False,
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.8,
                output_formats=['json'],
                verbose_logging=False
            ),
            
            'balanced': ParsingConfig(
                max_validation_attempts=2,
                sample_size_for_validation=75,
                enable_chain_of_thought=True,
                enable_fallback_strategies=False,
                enable_hypothesis_refinement=True,
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.85,
                output_formats=['json', 'txt'],
                verbose_logging=False
            ),
            
            'accurate': ParsingConfig(
                max_validation_attempts=3,
                sample_size_for_validation=150,
                enable_chain_of_thought=True,
                enable_fallback_strategies=False,
                enable_hypothesis_refinement=True,
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.9,
                output_formats=['json', 'txt', 'csv'],
                verbose_logging=True
            ),
            
            'minimal': ParsingConfig(
                max_validation_attempts=1,
                sample_size_for_validation=10,
                enable_chain_of_thought=False,
                enable_fallback_strategies=False,
                enable_hypothesis_refinement=False,
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.7,
                output_formats=['json'],
                verbose_logging=False
            ),
            
            'comprehensive': ParsingConfig(
                max_validation_attempts=5,
                sample_size_for_validation=200,
                enable_chain_of_thought=True,
                enable_fallback_strategies=False,  # Avoid noise
                enable_hypothesis_refinement=True,
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.95,
                output_formats=['json', 'txt', 'csv'],
                verbose_logging=True
            )
        }
    
    def get_preset_config(self, preset_name: str) -> Optional[ParsingConfig]:
        """
        Get a predefined configuration preset.
        
        Args:
            preset_name: Name of the preset ('fast', 'balanced', 'accurate', etc.)
            
        Returns:
            ParsingConfig or None if preset not found
        """
        return self.presets.get(preset_name.lower())
    
    def get_optimized_config(self, file_path: Path, priority: str = 'balanced') -> ParsingConfig:
        """
        Get optimized configuration for a specific file.
        
        Args:
            file_path: Path to SEGY file
            priority: 'speed', 'accuracy', or 'balanced'
            
        Returns:
            Optimized ParsingConfig
        """
        return self.tuner.auto_tune_config(file_path, priority)
    
    def get_config_for_target_time(self, file_path: Path, target_seconds: float) -> ParsingConfig:
        """
        Get configuration tuned for target processing time.
        
        Args:
            file_path: Path to SEGY file
            target_seconds: Target processing time in seconds
            
        Returns:
            Tuned ParsingConfig
        """
        return self.tuner.tune_config_for_target_time(file_path, target_seconds)
    
    def save_custom_config(self, name: str, config: ParsingConfig) -> None:
        """
        Save a custom configuration preset.
        
        Args:
            name: Name for the custom configuration
            config: ParsingConfig to save
        """
        config_file = self.config_dir / f"{name}.json"
        
        try:
            config_dict = asdict(config)
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save configuration '{name}': {e}")
    
    def load_custom_config(self, name: str) -> Optional[ParsingConfig]:
        """
        Load a custom configuration preset.
        
        Args:
            name: Name of the custom configuration
            
        Returns:
            ParsingConfig or None if not found
        """
        config_file = self.config_dir / f"{name}.json"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                    return ParsingConfig(**config_dict)
        except Exception as e:
            print(f"Warning: Could not load configuration '{name}': {e}")
        
        return None
    
    def list_available_configs(self) -> Dict[str, str]:
        """
        List all available configurations.
        
        Returns:
            Dictionary mapping config names to descriptions
        """
        configs = {
            'fast': 'Optimized for speed - minimal validation, basic output',
            'balanced': 'Good balance of speed and accuracy - recommended for most use cases',
            'accurate': 'Optimized for accuracy - thorough validation, comprehensive output',
            'minimal': 'Absolute minimum processing - fastest possible',
            'comprehensive': 'Maximum accuracy and detail - slowest but most thorough'
        }
        
        # Add custom configurations
        for config_file in self.config_dir.glob("*.json"):
            if config_file.stem not in configs:
                configs[config_file.stem] = 'Custom configuration'
        
        return configs
    
    def get_performance_recommendations(self, file_path: Path) -> Dict[str, Any]:
        """
        Get performance recommendations for a file.
        
        Args:
            file_path: Path to SEGY file
            
        Returns:
            Dictionary with recommendations
        """
        return self.optimizer.get_performance_recommendations(file_path)
    
    def _load_system_config(self) -> SystemConfiguration:
        """Load system configuration from file."""
        config_file = self.config_dir / "system.json"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                    return SystemConfiguration(**config_dict)
        except Exception:
            pass
        
        # Return default configuration
        return SystemConfiguration()
    
    def save_system_config(self) -> None:
        """Save system configuration to file."""
        config_file = self.config_dir / "system.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(asdict(self.system_config), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save system configuration: {e}")
    
    def cleanup_cache(self) -> None:
        """Clean up old cache files."""
        if self.system_config.cache_enabled:
            self.optimizer.cleanup_cache(self.system_config.cache_max_age_days)


def create_config_examples():
    """Create example configuration files for users."""
    config_manager = ConfigurationManager()
    
    # Example custom configurations
    examples = {
        'research': ParsingConfig(
            max_validation_attempts=4,
            sample_size_for_validation=100,
            enable_chain_of_thought=True,
            enable_fallback_strategies=False,
            enable_hypothesis_refinement=True,
            skip_validation_for_explicit=True,
            explicit_confidence_threshold=0.9,
            output_formats=['json', 'txt', 'csv'],
            verbose_logging=True
        ),
        
        'production': ParsingConfig(
            max_validation_attempts=2,
            sample_size_for_validation=50,
            enable_chain_of_thought=False,
            enable_fallback_strategies=False,
            enable_hypothesis_refinement=True,
            skip_validation_for_explicit=True,
            explicit_confidence_threshold=0.85,
            output_formats=['json'],
            verbose_logging=False
        ),
        
        'exploration': ParsingConfig(
            max_validation_attempts=1,
            sample_size_for_validation=30,
            enable_chain_of_thought=True,
            enable_fallback_strategies=False,
            enable_hypothesis_refinement=False,
            skip_validation_for_explicit=True,
            explicit_confidence_threshold=0.8,
            output_formats=['json', 'txt'],
            verbose_logging=False
        )
    }
    
    for name, config in examples.items():
        config_manager.save_custom_config(name, config)
    
    print("✅ Example configurations created:")
    for name, description in config_manager.list_available_configs().items():
        print(f"   {name}: {description}")


if __name__ == "__main__":
    create_config_examples()