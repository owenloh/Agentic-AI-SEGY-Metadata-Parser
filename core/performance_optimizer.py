"""
Performance Optimizer - Optimization strategies for the AI-Powered SEGY Metadata Parser.

This module provides performance optimizations including:
- Sampling strategies for large files
- Caching mechanisms for repeated operations
- Configuration tuning for different use cases
"""

import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

from enhanced_segy_parser import ParsingConfig


@dataclass
class PerformanceMetrics:
    """Performance metrics for parsing operations."""
    parsing_time: float
    validation_time: float
    geometric_extraction_time: float
    export_time: float
    total_time: float
    attributes_extracted: int
    attributes_validated: int
    file_size_mb: float
    sample_size_used: int


class PerformanceOptimizer:
    """Performance optimization strategies for SEGY parsing."""
    
    def __init__(self):
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.performance_history = {}
        
    def get_optimal_config_for_file_size(self, file_size_mb: float) -> ParsingConfig:
        """
        Get optimal configuration based on file size.
        
        Args:
            file_size_mb: File size in megabytes
            
        Returns:
            Optimized ParsingConfig
        """
        if file_size_mb < 50:  # Small files
            return ParsingConfig(
                max_validation_attempts=3,
                sample_size_for_validation=100,
                enable_chain_of_thought=True,
                enable_fallback_strategies=False,  # Avoid noise
                enable_hypothesis_refinement=True,
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.85,
                output_formats=['json', 'txt', 'csv'],
                verbose_logging=False
            )
        elif file_size_mb < 200:  # Medium files
            return ParsingConfig(
                max_validation_attempts=2,
                sample_size_for_validation=75,
                enable_chain_of_thought=True,
                enable_fallback_strategies=False,
                enable_hypothesis_refinement=True,
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.85,
                output_formats=['json', 'txt'],
                verbose_logging=False
            )
        else:  # Large files
            return ParsingConfig(
                max_validation_attempts=1,
                sample_size_for_validation=50,
                enable_chain_of_thought=False,  # Disable for speed
                enable_fallback_strategies=False,
                enable_hypothesis_refinement=False,  # Disable for speed
                skip_validation_for_explicit=True,
                explicit_confidence_threshold=0.8,  # Lower threshold for speed
                output_formats=['json'],  # Minimal output
                verbose_logging=False
            )
    
    def get_fast_config(self) -> ParsingConfig:
        """Get configuration optimized for speed."""
        return ParsingConfig(
            max_validation_attempts=1,
            sample_size_for_validation=25,
            enable_chain_of_thought=False,
            enable_fallback_strategies=False,
            enable_hypothesis_refinement=False,
            skip_validation_for_explicit=True,
            explicit_confidence_threshold=0.8,
            output_formats=['json'],
            verbose_logging=False
        )
    
    def get_accurate_config(self) -> ParsingConfig:
        """Get configuration optimized for accuracy."""
        return ParsingConfig(
            max_validation_attempts=3,
            sample_size_for_validation=150,
            enable_chain_of_thought=True,
            enable_fallback_strategies=False,  # Avoid noise
            enable_hypothesis_refinement=True,
            skip_validation_for_explicit=True,
            explicit_confidence_threshold=0.9,  # Higher threshold
            output_formats=['json', 'txt', 'csv'],
            verbose_logging=True
        )
    
    def get_balanced_config(self) -> ParsingConfig:
        """Get balanced configuration for speed and accuracy."""
        return ParsingConfig(
            max_validation_attempts=2,
            sample_size_for_validation=75,
            enable_chain_of_thought=True,
            enable_fallback_strategies=False,
            enable_hypothesis_refinement=True,
            skip_validation_for_explicit=True,
            explicit_confidence_threshold=0.85,
            output_formats=['json', 'txt'],
            verbose_logging=False
        )
    
    def calculate_optimal_sample_size(self, file_size_mb: float, target_time_seconds: float = 60) -> int:
        """
        Calculate optimal sample size based on file size and target processing time.
        
        Args:
            file_size_mb: File size in megabytes
            target_time_seconds: Target processing time in seconds
            
        Returns:
            Optimal sample size for validation
        """
        # Base sample size calculation
        if file_size_mb < 10:
            base_sample = 100
        elif file_size_mb < 50:
            base_sample = 75
        elif file_size_mb < 200:
            base_sample = 50
        else:
            base_sample = 25
        
        # Adjust based on target time
        if target_time_seconds < 30:  # Fast processing
            return max(base_sample // 2, 10)
        elif target_time_seconds > 120:  # Thorough processing
            return min(base_sample * 2, 200)
        else:
            return base_sample
    
    def get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for a SEGY file."""
        # Use file path, size, and modification time for cache key
        stat = file_path.stat()
        cache_data = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def cache_textual_header(self, file_path: Path, textual_header: str) -> None:
        """Cache textual header to avoid re-extraction."""
        cache_key = self.get_cache_key(file_path)
        cache_file = self.cache_dir / f"header_{cache_key}.txt"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(textual_header)
        except Exception:
            pass  # Ignore cache failures
    
    def get_cached_textual_header(self, file_path: Path) -> Optional[str]:
        """Get cached textual header if available."""
        cache_key = self.get_cache_key(file_path)
        cache_file = self.cache_dir / f"header_{cache_key}.txt"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception:
            pass
        
        return None
    
    def record_performance_metrics(self, file_path: Path, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for analysis."""
        cache_key = self.get_cache_key(file_path)
        self.performance_history[cache_key] = {
            'file_path': str(file_path),
            'file_size_mb': metrics.file_size_mb,
            'total_time': metrics.total_time,
            'attributes_extracted': metrics.attributes_extracted,
            'attributes_validated': metrics.attributes_validated,
            'sample_size_used': metrics.sample_size_used,
            'timestamp': time.time()
        }
        
        # Save to disk
        try:
            history_file = self.cache_dir / "performance_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception:
            pass  # Ignore save failures
    
    def get_performance_recommendations(self, file_path: Path) -> Dict[str, Any]:
        """Get performance recommendations based on historical data."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        recommendations = {
            'optimal_config': self.get_optimal_config_for_file_size(file_size_mb),
            'estimated_time_fast': self._estimate_processing_time(file_size_mb, 'fast'),
            'estimated_time_balanced': self._estimate_processing_time(file_size_mb, 'balanced'),
            'estimated_time_accurate': self._estimate_processing_time(file_size_mb, 'accurate'),
            'recommended_sample_size': self.calculate_optimal_sample_size(file_size_mb),
            'file_size_category': self._get_file_size_category(file_size_mb)
        }
        
        return recommendations
    
    def _estimate_processing_time(self, file_size_mb: float, mode: str) -> float:
        """Estimate processing time based on file size and mode."""
        # Base time estimates (in seconds)
        base_times = {
            'fast': 0.5,      # 0.5 seconds per MB
            'balanced': 1.0,  # 1 second per MB
            'accurate': 2.0   # 2 seconds per MB
        }
        
        base_time = base_times.get(mode, 1.0)
        
        # Scale with file size (with diminishing returns for large files)
        if file_size_mb < 10:
            time_factor = 1.0
        elif file_size_mb < 50:
            time_factor = 0.8
        elif file_size_mb < 200:
            time_factor = 0.6
        else:
            time_factor = 0.4
        
        return file_size_mb * base_time * time_factor
    
    def _get_file_size_category(self, file_size_mb: float) -> str:
        """Get file size category."""
        if file_size_mb < 50:
            return "small"
        elif file_size_mb < 200:
            return "medium"
        else:
            return "large"
    
    def cleanup_cache(self, max_age_days: int = 30) -> None:
        """Clean up old cache files."""
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file():
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        cache_file.unlink()
        except Exception:
            pass  # Ignore cleanup failures


class ConfigurationTuner:
    """Automatic configuration tuning based on performance feedback."""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        
    def tune_config_for_target_time(self, file_path: Path, target_time_seconds: float) -> ParsingConfig:
        """
        Tune configuration to meet target processing time.
        
        Args:
            file_path: Path to SEGY file
            target_time_seconds: Target processing time
            
        Returns:
            Tuned ParsingConfig
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Start with balanced config
        config = self.optimizer.get_balanced_config()
        
        # Adjust based on target time
        if target_time_seconds < 30:  # Fast target
            config.max_validation_attempts = 1
            config.sample_size_for_validation = 25
            config.enable_chain_of_thought = False
            config.enable_hypothesis_refinement = False
            config.output_formats = ['json']
            
        elif target_time_seconds < 60:  # Medium target
            config.max_validation_attempts = 2
            config.sample_size_for_validation = 50
            config.enable_chain_of_thought = True
            config.enable_hypothesis_refinement = True
            config.output_formats = ['json', 'txt']
            
        else:  # Thorough target
            config.max_validation_attempts = 3
            config.sample_size_for_validation = 100
            config.enable_chain_of_thought = True
            config.enable_hypothesis_refinement = True
            config.output_formats = ['json', 'txt', 'csv']
        
        # Adjust sample size based on file size
        config.sample_size_for_validation = self.optimizer.calculate_optimal_sample_size(
            file_size_mb, target_time_seconds
        )
        
        return config
    
    def auto_tune_config(self, file_path: Path, priority: str = 'balanced') -> ParsingConfig:
        """
        Automatically tune configuration based on priority.
        
        Args:
            file_path: Path to SEGY file
            priority: 'speed', 'accuracy', or 'balanced'
            
        Returns:
            Auto-tuned ParsingConfig
        """
        if priority == 'speed':
            return self.optimizer.get_fast_config()
        elif priority == 'accuracy':
            return self.optimizer.get_accurate_config()
        else:
            return self.optimizer.get_balanced_config()