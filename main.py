#!/usr/bin/env python3
"""
AI-Powered SEGY Metadata Parser - Main Interface

This is the primary entry point for the AI-Powered SEGY Metadata Parser system.
Choose from two main functions:
1. Parse SEGY files to extract attributes, geometry, and revision info
2. View SEGY textual headers to understand file structure

Usage:
    python main.py                    # Interactive menu
    python main.py parse file.sgy    # Direct parsing
    python main.py view file.sgy     # View header
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
def load_env_file():
    env_file = Path(".env")
    if env_file.exists():
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


def show_banner():
    """Display the AI-Powered SEGY Metadata Parser banner."""
    print("=" * 80)
    print("🚀 AI-POWERED SEGY METADATA PARSER")
    print("   AI-Powered Seismic Data Analysis")
    print("=" * 80)
    print("📊 Extract: SEGY revision, attribute mappings, geometric data")
    print("🤖 Powered by: Advanced LLM analysis with smart validation")
    print("⚡ Modes: Fast (15s) | Balanced (30s) | Accurate (60s)")
    print("=" * 80)


def show_menu():
    """Display the main menu."""
    print("\n🎯 What would you like to do?")
    print()
    print("1. 📋 Parse SEGY File")
    print("   Extract attributes, geometry, and revision information")
    print()
    print("2. 👁️  View SEGY Header")
    print("   Display textual header content for analysis")
    print()
    print("3. ❓ Help & Documentation")
    print("   View usage examples and configuration options")
    print()
    print("4. 🚪 Exit")
    print()


def get_segy_files_from_input():
    """Get SEGY file paths from user input or segypaths.txt"""
    file_paths = []
    
    # Get file path from user
    file_path = input("📁 Enter SEGY file path (or press Enter to use segypaths.txt): ").strip()
    
    if not file_path:
        # Check for segypaths.txt
        segypaths_file = Path("segypaths.txt")
        if segypaths_file.exists():
            with open(segypaths_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if Path(line).exists():
                            file_paths.append(line)
                        else:
                            print(f"⚠️  File not found: {line}")
            
            if file_paths:
                if len(file_paths) == 1:
                    print(f"✅ Using file from segypaths.txt: {Path(file_paths[0]).name}")
                else:
                    print(f"✅ Found {len(file_paths)} files in segypaths.txt:")
                    for i, fp in enumerate(file_paths, 1):
                        print(f"   {i}. {Path(fp).name}")
            else:
                print("❌ No valid SEGY file paths found in segypaths.txt")
                print("Please add valid SEGY file paths to segypaths.txt or provide one directly.")
                return []
        else:
            print("❌ No segypaths.txt found. Please provide a file path.")
            return []
    else:
        # Single file from user input
        if Path(file_path).exists():
            file_paths = [file_path]
        else:
            print(f"❌ File not found: {file_path}")
            return []
    
    return file_paths


def parse_single_segy(file_path, config, output_path, verbose=True):
    """Parse a single SEGY file with configurable verbosity."""
    from enhanced_segy_parser import SEGYHeaderParser
    from datetime import datetime
    
    if verbose:
        print(f"\n🚀 Processing: {Path(file_path).name}")
        print("-" * 50)
    else:
        print(f"🔄 {Path(file_path).name}...", end=" ", flush=True)
    
    try:
        # Initialize parser
        parser = SEGYHeaderParser(config)
        
        # Parse the file
        start_time = datetime.now()
        result = parser.parse_segy_file(Path(file_path), output_path)
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        validated_count = len([a for a in result.attributes if a.validation_status == "validated"])
        
        # Display results based on verbosity
        if verbose:
            print(f"✅ Completed: {Path(file_path).name}")
            print(f"⏱️  Time: {processing_time:.2f}s")
            print(f"📊 Revision: {result.revision_info.revision}")
            print(f"🎯 Attributes: {len(result.attributes)}")
            print(f"✅ Validated: {validated_count}/{len(result.attributes)}")
        else:
            print(f"✅ {processing_time:.1f}s ({len(result.attributes)} attrs, {validated_count} validated)")
        
        return True, processing_time, len(result.attributes), validated_count
        
    except Exception as e:
        if verbose:
            print(f"❌ Error processing {Path(file_path).name}: {e}")
        else:
            print(f"❌ Error: {e}")
        return False, 0, 0, 0


def parse_segy_interactive():
    """Interactive SEGY parsing with batch support."""
    from config_system import ConfigurationManager
    
    print("\n📋 SEGY FILE PARSING")
    print("=" * 50)
    
    # Get file paths
    file_paths = get_segy_files_from_input()
    if not file_paths:
        return
    
    # Get configuration
    print("\n⚙️  Choose processing mode:")
    print("1. Fast (10-15s) - Quick analysis")
    print("2. Balanced (15-30s) - Recommended")
    print("3. Accurate (30-60s) - Thorough analysis")
    
    config_choice = input("Enter choice (1-3, default=2): ").strip() or "2"
    config_map = {"1": "fast", "2": "balanced", "3": "accurate"}
    config_name = config_map.get(config_choice, "balanced")
    
    # Get output directory
    output_dir = input("📂 Output directory (default=./output): ").strip() or "./output"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    config_manager = ConfigurationManager()
    config = config_manager.get_preset_config(config_name)
    
    if not config:
        print(f"❌ Error: Unknown configuration: {config_name}")
        return
    
    # Process files
    total_files = len(file_paths)
    if total_files == 1:
        print(f"\n🚀 Starting SEGY analysis...")
        print(f"📁 File: {Path(file_paths[0]).name}")
        print(f"⚙️  Mode: {config_name}")
        print(f"📂 Output: {output_path}")
    else:
        print(f"\n🚀 Starting batch SEGY analysis...")
        print(f"📁 Files: {total_files} files")
        print(f"⚙️  Mode: {config_name}")
        print(f"📂 Output: {output_path}")
    
    # Process all files
    successful = 0
    failed = 0
    total_time = 0
    total_attributes = 0
    total_validated = 0
    
    from datetime import datetime
    batch_start = datetime.now()
    
    for i, file_path in enumerate(file_paths, 1):
        if total_files > 1:
            print(f"\n📊 Processing file {i}/{total_files}")
        
        success, proc_time, attrs, validated = parse_single_segy(file_path, config, output_path)
        
        if success:
            successful += 1
            total_time += proc_time
            total_attributes += attrs
            total_validated += validated
        else:
            failed += 1
    
    batch_end = datetime.now()
    batch_time = (batch_end - batch_start).total_seconds()
    
    # Display batch summary
    print("\n" + "=" * 60)
    print("🎉 BATCH PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"📊 Files processed: {successful}/{total_files}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {batch_time:.2f} seconds")
    if successful > 0:
        print(f"⚡ Average time per file: {total_time/successful:.2f} seconds")
        print(f"🎯 Total attributes found: {total_attributes}")
        print(f"✅ Total validated: {total_validated}")
    
    # Show output files
    print(f"\n📁 Output files in: {output_path}")
    output_files = list(output_path.glob("*"))
    if output_files:
        for output_file in sorted(output_files):
            if output_file.is_file():
                file_size = output_file.stat().st_size
                print(f"   📄 {output_file.name} ({file_size:,} bytes)")
    
    print(f"\n🎉 Analysis complete! Check the output directory for all results.")


def view_single_header(file_path):
    """View header for a single SEGY file."""
    import segyio
    
    try:
        print(f"\n📁 Reading SEGY file: {Path(file_path).name}")
        print(f"📊 File size: {Path(file_path).stat().st_size:,} bytes")
        print("=" * 80)
        
        with segyio.open(str(file_path), "r", ignore_geometry=True) as segy_file:
            # Get the textual header
            textual_header = segy_file.text[0]
            
            print("TEXTUAL HEADER CONTENT:")
            print("-" * 80)
            
            # Convert bytes to string and split into 80-character lines
            header_text = textual_header.decode('ascii', errors='replace')
            
            # Print each 80-character line with line numbers
            line_count = 0
            for i in range(0, len(header_text), 80):
                line = header_text[i:i+80].rstrip()
                line_count += 1
                print(f"C{line_count:02d}: {line}")
            
            print("-" * 80)
            print(f"Total lines: {line_count}")
            
            # Show binary header info
            print("\nBINARY HEADER INFO:")
            print("-" * 80)
            print(f"Sample interval: {segy_file.bin[segyio.BinField.Interval]} microseconds")
            print(f"Samples per trace: {segy_file.bin[segyio.BinField.Samples]}")
            print(f"Number of traces: {segy_file.tracecount}")
            print(f"Data format: {segy_file.bin[segyio.BinField.Format]}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error reading SEGY file {Path(file_path).name}: {e}")
        return False


def view_header_interactive():
    """Interactive header viewing with batch support."""
    print("\n👁️  SEGY HEADER VIEWER")
    print("=" * 50)
    
    # Get file paths
    file_paths = get_segy_files_from_input()
    if not file_paths:
        return
    
    total_files = len(file_paths)
    successful = 0
    
    if total_files == 1:
        print(f"\n👁️  Viewing header for: {Path(file_paths[0]).name}")
    else:
        print(f"\n👁️  Viewing headers for {total_files} files")
        
        # Ask if user wants to view all or select specific files
        if total_files > 3:
            view_all = input(f"\n🤔 View all {total_files} headers? (y/n, default=y): ").strip().lower()
            if view_all in ['n', 'no']:
                print("\n📋 Available files:")
                for i, fp in enumerate(file_paths, 1):
                    print(f"   {i}. {Path(fp).name}")
                
                selection = input("\nEnter file numbers to view (e.g., 1,3,5 or 'all'): ").strip()
                if selection.lower() != 'all':
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        file_paths = [file_paths[i] for i in indices if 0 <= i < len(file_paths)]
                        total_files = len(file_paths)
                    except (ValueError, IndexError):
                        print("❌ Invalid selection. Viewing all files.")
    
    # View headers
    for i, file_path in enumerate(file_paths, 1):
        if total_files > 1:
            print(f"\n{'='*20} FILE {i}/{total_files} {'='*20}")
        
        if view_single_header(file_path):
            successful += 1
        
        # For multiple files, ask if user wants to continue
        if total_files > 1 and i < total_files:
            continue_viewing = input(f"\n⏭️  Continue to next file? (y/n, default=y): ").strip().lower()
            if continue_viewing in ['n', 'no']:
                break
    
    print(f"\n✅ Header viewing completed! Viewed {successful}/{len(file_paths)} files successfully.")


def show_help():
    """Show help and documentation."""
    print("\n❓ HELP & DOCUMENTATION")
    print("=" * 50)
    print()
    print("📚 Available Documentation:")
    print("   • README.md - Complete project overview")
    print("   • QUICKSTART.md - Get started in 3 steps")
    print("   • USER_INSTRUCTIONS.md - Comprehensive guide")
    print()
    print("🚀 Quick Examples:")
    print("   # Parse a SEGY file with balanced settings")
    print("   python main.py parse survey.sgy")
    print()
    print("   # View SEGY textual header")
    print("   python main.py view survey.sgy")
    print()
    print("   # Direct CLI usage")
    print("   python parse_segy.py survey.sgy --config fast")
    print("   python print_textual_header.py survey.sgy")
    print()
    print("⚙️  Configuration Options:")
    print("   • fast: Quick analysis (10-15s)")
    print("   • balanced: Recommended balance (15-30s)")
    print("   • accurate: Thorough analysis (30-60s)")
    print()
    print("📁 Setup Requirements:")
    print("   1. Create .env file with API key:")
    print("      GEMINI_API_KEY=your_api_key_here")
    print("   2. Install dependencies: pip install -r requirements.txt")
    print("   3. Run: python main.py")
    print()


def main():
    """Main function with command-line and interactive modes."""
    
    # Handle direct command-line usage
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "parse" and len(sys.argv) > 2:
            # Direct parsing: python main.py parse file.sgy
            from parse_segy import main as parse_main
            sys.argv = ["parse_segy.py"] + sys.argv[2:]  # Pass remaining args
            parse_main()
            return
            
        elif command == "view" and len(sys.argv) > 2:
            # Direct header viewing: python main.py view file.sgy
            from print_textual_header import main as header_main
            sys.argv = ["print_textual_header.py"] + sys.argv[2:]  # Pass remaining args
            header_main()
            return
            
        elif command in ["help", "--help", "-h"]:
            show_banner()
            show_help()
            return
    
    # Interactive mode
    show_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                parse_segy_interactive()
                
            elif choice == "2":
                view_header_interactive()
                
            elif choice == "3":
                show_help()
                
            elif choice == "4":
                print("\n👋 Thank you for using AI-Powered SEGY Metadata Parser!")
                print("🌟 Star us on GitHub if this helped your seismic analysis!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        
        # Ask if user wants to continue
        if choice in ["1", "2"]:
            continue_choice = input("\n🔄 Would you like to do something else? (y/n): ").strip().lower()
            if continue_choice not in ["y", "yes", ""]:
                print("\n👋 Thank you for using AI-Powered SEGY Metadata Parser!")
                break


if __name__ == "__main__":
    main()