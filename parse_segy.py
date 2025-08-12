#!/usr/bin/env python3
"""
Simple CLI script for the AI-Powered SEGY Metadata Parser.

Usage:
    python parse_segy.py path/to/file.sgy
    python parse_segy.py path/to/file.sgy --output-dir ./results
    python parse_segy.py path/to/file.sgy --config fast
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Load environment variables
def load_env_file():
    env_file = Path(".env")
    if env_file.exists():
        import os
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env_file()

# Configure logging to suppress debug messages
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

# Suppress noisy loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('requests.packages.urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

from enhanced_segy_parser import SEGYHeaderParser, ParsingConfig
from config_system import ConfigurationManager


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered SEGY Metadata Parser - Extract revision, geometry, and attribute mappings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_segy.py survey.sgy
  python parse_segy.py survey.sgy --output-dir ./results
  python parse_segy.py survey.sgy --config fast
  python parse_segy.py survey.sgy --config accurate --verbose

Configuration Options:
  fast      - Quick processing with minimal validation
  balanced  - Good balance of speed and accuracy (default)
  accurate  - Thorough analysis with comprehensive validation
        """
    )
    
    parser.add_argument(
        'segy_file',
        type=str,
        help='Path to the SEGY file to parse'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        choices=['fast', 'balanced', 'accurate', 'minimal', 'comprehensive'],
        default='balanced',
        help='Configuration preset (default: balanced)'
    )
    
    parser.add_argument(
        '--formats', '-f',
        type=str,
        nargs='+',
        choices=['json', 'txt', 'csv'],
        default=['json', 'txt', 'csv'],
        help='Output formats to generate (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--target-time', '-t',
        type=float,
        help='Target processing time in seconds (overrides config preset)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    segy_file = Path(args.segy_file)
    if not segy_file.exists():
        print(f"❌ Error: SEGY file not found: {segy_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AI-POWERED SEGY METADATA PARSER")
    print("=" * 80)
    print(f"📁 Input file: {segy_file}")
    print(f"📂 Output directory: {output_dir}")
    print(f"⚙️  Configuration: {args.config}")
    print(f"📄 Output formats: {', '.join(args.formats)}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Get configuration
        config_manager = ConfigurationManager()
        
        if args.target_time:
            print(f"🎯 Using target time optimization: {args.target_time}s")
            config = config_manager.get_config_for_target_time(segy_file, args.target_time)
        else:
            config = config_manager.get_preset_config(args.config)
            if not config:
                print(f"❌ Error: Unknown configuration preset: {args.config}")
                sys.exit(1)
        
        # Override output formats and verbose settings
        config.output_formats = args.formats
        config.verbose_logging = args.verbose
        
        # Initialize parser
        print("🏗️  Initializing AI-Powered SEGY Metadata Parser...")
        segy_parser = SEGYHeaderParser(config)
        
        # Parse the file
        print("🚀 Starting SEGY analysis...")
        start_time = datetime.now()
        
        result = segy_parser.parse_segy_file(segy_file, output_dir)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Display results summary
        print("\n" + "=" * 80)
        print("PARSING RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"✅ Processing completed successfully!")
        print(f"⏱️  Total time: {processing_time:.2f} seconds")
        print(f"📊 SEGY revision: {result.revision_info.revision}")
        print(f"🎯 Attributes found: {len(result.attributes)}")
        
        validated_count = len([a for a in result.attributes if a.validation_status == "validated"])
        print(f"✅ Attributes validated: {validated_count}/{len(result.attributes)}")
        
        print(f"🗺️  Geometric coordinates: {len(result.geometric_info.world_coordinates)}")
        print(f"📐 Inline/crossline mappings: {len(result.geometric_info.inline_crossline)}")
        
        # Show confidence distribution
        confidence_summary = result.confidence_summary
        print(f"📈 Confidence distribution:")
        for level, count in confidence_summary.items():
            print(f"   {level.capitalize()}: {count} attributes")
        
        # List output files
        print(f"\n📁 Output files created in: {output_dir}")
        for output_file in output_dir.glob(f"{segy_file.stem}*"):
            file_size = output_file.stat().st_size
            print(f"   📄 {output_file.name} ({file_size:,} bytes)")
        
        # Show top attributes
        print(f"\n🎯 Top Attribute Mappings:")
        sorted_attrs = sorted(result.attributes, key=lambda x: x.confidence, reverse=True)
        for i, attr in enumerate(sorted_attrs[:5], 1):
            status_icon = "✅" if attr.validation_status == "validated" else "⚠️"
            print(f"   {i}. {status_icon} {attr.attribute_name}")
            print(f"      Bytes: {attr.byte_start}-{attr.byte_end} | Confidence: {attr.confidence:.3f}")
        
        if len(sorted_attrs) > 5:
            print(f"   ... and {len(sorted_attrs) - 5} more attributes")
        
        print(f"\n🎉 Analysis complete! Check the output files for detailed results.")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()