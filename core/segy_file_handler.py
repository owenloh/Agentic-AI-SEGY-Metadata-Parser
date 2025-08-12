"""
SEGYFileHandler - Central interface to segyio for all SEGY file operations.
"""

import segyio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from models.data_models import SEGYFileInfo


class SEGYFileHandler:
    """Central interface to segyio for all SEGY file operations"""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.segy_file: Optional[segyio.SegyFile] = None
        self.file_info: Optional[SEGYFileInfo] = None
        
    def open_file(self) -> bool:
        """Opens SEGY file using segyio and extracts basic info"""
        try:
            # Try opening with strict=False to handle non-standard files
            self.segy_file = segyio.open(str(self.filepath), "r", strict=False)
            self._extract_file_info()
            return True
        except Exception as e:
            print(f"Error opening SEGY file {self.filepath}: {e}")
            # Try with ignore_geometry=True for problematic files
            try:
                self.segy_file = segyio.open(str(self.filepath), "r", strict=False, ignore_geometry=True)
                self._extract_file_info()
                return True
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                return False
    
    def close_file(self) -> None:
        """Close the SEGY file"""
        if self.segy_file:
            self.segy_file.close()
            self.segy_file = None
    
    def _extract_file_info(self) -> None:
        """Extract basic file information"""
        if not self.segy_file:
            return
            
        try:
            self.file_info = SEGYFileInfo(
                filepath=str(self.filepath),
                file_size=self.filepath.stat().st_size,
                trace_count=len(self.segy_file.trace),
                sample_count=len(self.segy_file.samples),
                sample_interval=self.segy_file.bin[segyio.BinField.Interval] / 1000.0,  # Convert to ms
                format_code=self.segy_file.bin[segyio.BinField.Format],
                has_textual_headers=True,  # segyio always provides textual headers
                has_binary_header=True     # segyio always provides binary header
            )
        except Exception as e:
            print(f"Error extracting file info: {e}")
    
    def get_textual_headers(self) -> List[str]:
        """Extract all textual headers using segyio"""
        if not self.segy_file:
            return []
        
        try:
            # Get textual headers - segyio provides them as bytes
            textual_headers = []
            for i, header in enumerate(self.segy_file.text):
                # Convert bytes to string, handling different encodings
                try:
                    header_str = header.decode('ascii')
                except UnicodeDecodeError:
                    try:
                        header_str = header.decode('ebcdic-cp-us')
                    except UnicodeDecodeError:
                        header_str = header.decode('utf-8', errors='ignore')
                
                textual_headers.append(header_str)
            
            return textual_headers
        except Exception as e:
            print(f"Error reading textual headers: {e}")
            return []
    
    def get_binary_header(self) -> Dict[str, Any]:
        """Extract binary header information using segyio"""
        if not self.segy_file:
            return {}
        
        try:
            binary_header = {}
            
            # Extract key binary header fields
            bin_header = self.segy_file.bin
            
            # Common binary header fields
            binary_header.update({
                'job_id': bin_header.get(segyio.BinField.JobID, 0),
                'line_number': bin_header.get(segyio.BinField.LineNumber, 0),
                'reel_number': bin_header.get(segyio.BinField.ReelNumber, 0),
                'traces_per_ensemble': bin_header.get(segyio.BinField.TracesPerEnsemble, 0),
                'auxiliary_traces_per_ensemble': bin_header.get(segyio.BinField.AuxTracesPerEnsemble, 0),
                'sample_interval': bin_header.get(segyio.BinField.Interval, 0),
                'samples_per_trace': bin_header.get(segyio.BinField.Samples, 0),
                'data_sample_format': bin_header.get(segyio.BinField.Format, 0),
                'ensemble_fold': bin_header.get(segyio.BinField.EnsembleFold, 0),
                'trace_sorting_code': bin_header.get(segyio.BinField.SortingCode, 0),
                'vertical_sum_code': bin_header.get(segyio.BinField.VerticalSumCode, 0),
                'sweep_frequency_start': bin_header.get(segyio.BinField.SweepFrequencyStart, 0),
                'sweep_frequency_end': bin_header.get(segyio.BinField.SweepFrequencyEnd, 0),
                'sweep_length': bin_header.get(segyio.BinField.SweepLength, 0),
                'sweep_type_code': bin_header.get(segyio.BinField.SweepTypeCode, 0),
                'measurement_system': bin_header.get(segyio.BinField.MeasurementSystem, 0),
                'impulse_signal_polarity': bin_header.get(segyio.BinField.ImpulseSignalPolarity, 0),
                'vibratory_polarity_code': bin_header.get(segyio.BinField.VibratoryPolarityCode, 0),
                'segy_revision': bin_header.get(segyio.BinField.SEGYRevision, 0),
                'trace_value_measurement_unit': bin_header.get(segyio.BinField.TraceValueMeasurementUnit, 0),
                'transduction_constant_mantissa': bin_header.get(segyio.BinField.TransductionConstantMantissa, 0),
                'transduction_constant_exponent': bin_header.get(segyio.BinField.TransductionConstantExponent, 0),
                'transduction_units': bin_header.get(segyio.BinField.TransductionUnits, 0)
            })
            
            return binary_header
            
        except Exception as e:
            print(f"Error reading binary header: {e}")
            return {}
    
    def get_trace_sample(self, trace_indices: List[int], byte_range: Tuple[int, int]) -> np.ndarray:
        """Extract trace header data from specified byte range for given traces"""
        if not self.segy_file:
            return np.array([])
        
        try:
            byte_start, byte_end = byte_range
            data = []
            
            for trace_idx in trace_indices:
                if trace_idx < len(self.segy_file.trace):
                    # Get trace header as raw bytes
                    trace_header = self.segy_file.header[trace_idx]
                    
                    # Extract bytes from the specified range
                    # Note: segyio trace headers are accessed by field, but we need raw bytes
                    # This is a simplified approach - in practice, you'd need to handle
                    # the binary structure more carefully
                    header_bytes = bytes(trace_header)
                    if byte_end <= len(header_bytes):
                        raw_bytes = header_bytes[byte_start:byte_end]
                        # Convert bytes to numeric value (assuming big-endian int32)
                        if len(raw_bytes) == 4:
                            value = int.from_bytes(raw_bytes, byteorder='big', signed=True)
                            data.append(value)
                        elif len(raw_bytes) == 2:
                            value = int.from_bytes(raw_bytes, byteorder='big', signed=True)
                            data.append(value)
            
            return np.array(data)
            
        except Exception as e:
            print(f"Error extracting trace sample data: {e}")
            return np.array([])
    
    def get_trace_header_sample(self, trace_indices: List[int], byte_range: Tuple[int, int]) -> np.ndarray:
        """Extract trace header samples using segyio trace header access"""
        if not self.segy_file:
            return np.array([])
        
        try:
            # This is a more direct approach using segyio's trace header fields
            # Map common byte ranges to segyio TraceField enums
            byte_start, byte_end = byte_range
            data = []
            
            # Common trace header field mappings
            field_mapping = {
                (1, 4): segyio.TraceField.TRACE_SEQUENCE_LINE,
                (5, 8): segyio.TraceField.TRACE_SEQUENCE_FILE,
                (9, 12): segyio.TraceField.FieldRecord,
                (13, 16): segyio.TraceField.TraceNumber,
                (17, 20): segyio.TraceField.EnergySourcePoint,
                (21, 24): segyio.TraceField.CDP,
                (25, 28): segyio.TraceField.CDP_TRACE,
                (29, 30): segyio.TraceField.TraceIdentificationCode,
                (73, 76): segyio.TraceField.SourceX,
                (77, 80): segyio.TraceField.SourceY,
                (81, 84): segyio.TraceField.GroupX,
                (85, 88): segyio.TraceField.GroupY,
                (181, 184): segyio.TraceField.CDP_X,
                (185, 188): segyio.TraceField.CDP_Y,
                (189, 192): segyio.TraceField.INLINE_3D,
                (193, 196): segyio.TraceField.CROSSLINE_3D
            }
            
            # Check if we have a direct mapping
            trace_field = field_mapping.get((byte_start, byte_end))
            
            if trace_field:
                for trace_idx in trace_indices:
                    if trace_idx < len(self.segy_file.trace):
                        value = self.segy_file.header[trace_idx][trace_field]
                        data.append(value)
            else:
                # Fallback to raw byte extraction
                return self.get_trace_sample(trace_indices, byte_range)
            
            return np.array(data)
            
        except Exception as e:
            print(f"Error extracting trace header sample: {e}")
            return np.array([])
    
    def get_sample_trace_indices(self, sample_size: int = 100) -> List[int]:
        """Get a representative sample of trace indices for validation"""
        if not self.segy_file or not self.file_info:
            return []
        
        total_traces = self.file_info.trace_count
        if total_traces <= sample_size:
            return list(range(total_traces))
        
        # Get evenly distributed sample
        step = total_traces // sample_size
        return list(range(0, total_traces, step))[:sample_size]
    
    def __enter__(self):
        """Context manager entry"""
        self.open_file()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_file()