#!/usr/bin/env python3
"""
SEGY Header Viewer - Display SEGY textual headers in a readable format.

This utility script extracts and displays the textual header content from SEGY files
to help understand the data structure and what the AI-Powered SEGY Metadata Parser analyzes.

Usage:
    python print_textual_header.py path/to/file.sgy
    python print_textual_header.py  # Uses segypaths.txt
"""

import sys
import segyio
from pathlib import Path
from datetime import datetime


def print_textual_header(segy_file_path: Path, show_line_numbers: bool = True):
    """
    Print the textual header from a SEGY file in a readable format.
    
    Args:
        segy_file_path: Path to the SEGY file
        show_line_numbers: Whether to show line numbers (C01, C02, etc.)
    """
    print("=" * 100)
    print(f"SEGY TEXTUAL HEADER: {segy_file_path.name}")
    print("=" * 100)
    print(f"File: {segy_file_path}")
    print(f"Size: {segy_file_path.stat().st_size:,} bytes")
    print(f"Extracted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    try:
        with segyio.open(str(segy_file_path), "r", ignore_geometry=True) as segy_file:
            # Get the textual header
            textual_header = segy_file.text[0]
            
            print("TEXTUAL HEADER CONTENT:")
            print("-" * 100)
            
            # Convert bytes to string and split into 80-character lines
            header_text = textual_header.decode('ascii', errors='replace')
            
            # Print each 80-character line with line numbers
            line_count = 0
            for i in range(0, len(header_text), 80):
                line = header_text[i:i+80].rstrip()
                line_count += 1
                
                if show_line_numbers:
                    print(f"C{line_count:02d}: {line}")
                else:
                    print(line)
            
            print("-" * 100)
            print(f"Total lines: {line_count}")
            
            # Show binary header info
            print("\nBINARY HEADER INFO:")
            print("-" * 100)
            print(f"Sample interval: {segy_file.bin[segyio.BinField.Interval]} microseconds")
            print(f"Samples per trace: {segy_file.bin[segyio.BinField.Samples]}")
            print(f"Number of traces: {segy_file.tracecount}")
            print(f"Data format: {segy_file.bin[segyio.BinField.Format]}")
            
    except Exception as e:
        print(f"❌ Error reading SEGY file: {e}")
        return False
    
    return True


def get_segy_file_paths():
    """Get SEGY file paths from command line or segypaths.txt"""
    
    # Check command line argument
    if len(sys.argv) > 1:
        segy_path = Path(sys.argv[1])
        if segy_path.exists():
            return [segy_path]
        else:
            print(f"❌ File not found: {segy_path}")
            return []
    
    # Try segypaths.txt
    segypaths_file = Path("segypaths.txt")
    if segypaths_file.exists():
        file_paths = []
        try:
            with open(segypaths_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        segy_path = Path(line)
                        if segy_path.exists():
                            file_paths.append(segy_path)
                        else:
                            print(f"⚠️  File in segypaths.txt not found: {segy_path}")
            
            if file_paths:
                return file_paths
            else:
                print("❌ No valid SEGY files found in segypaths.txt")
                return []
                
        except Exception as e:
            print(f"❌ Error reading segypaths.txt: {e}")
            return []
    
    print("❌ No SEGY file specified. Usage:")
    print("   python print_textual_header.py path/to/file.sgy")
    print("   or create segypaths.txt with file paths")
    return []


def main():
    """Main function to display SEGY textual headers (supports batch processing)."""
    
    # Get the SEGY file paths
    segy_file_paths = get_segy_file_paths()
    if not segy_file_paths:
        sys.exit(1)
    
    total_files = len(segy_file_paths)
    successful = 0
    
    if total_files == 1:
        print(f"👁️  Displaying header for: {segy_file_paths[0].name}")
        if print_textual_header(segy_file_paths[0]):
            successful = 1
            print("\n✅ Header extraction completed successfully!")
    else:
        print(f"👁️  Found {total_files} SEGY files in segypaths.txt")
        print("🔄 Processing all files...")
        
        for i, segy_file_path in enumerate(segy_file_paths, 1):
            print(f"\n{'='*20} FILE {i}/{total_files} {'='*20}")
            
            if print_textual_header(segy_file_path):
                successful += 1
            
            # For multiple files, add separator
            if i < total_files:
                print("\n" + "="*80)
    
    print(f"\n✅ Header display completed! Processed {successful}/{total_files} files successfully!")
    
    if successful == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()