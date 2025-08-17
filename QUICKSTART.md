# 🚀 Quick Start - Main Function

## ⚡ Get Running in 60 Seconds

main.py runs the entire user interface with menu options, but setup the AI/LLM below: 

### 1. Setup (30 seconds)
```bash
# Copy the complete .env template (see below)
# Replace your_gemini_api_key_here with your actual API key

# Put your SEGY file path in segypaths.txt (optional)
echo "path/to/your/file.sgy" > segypaths.txt
```

**Get Gemini API Key (Free):**
1. Go to https://aistudio.google.com/
2. Create account and get API key
3. Copy the complete .env template from section "Complete .env Template:" below
4. Replace `your_gemini_api_key_here` with your actual key

### 2. Run Main Function (30 seconds)
```bash
python main.py
```

**That's it!** The interactive menu guides you through everything.

### **Direct CLI Usage** (Skip Interactive Menu)
```bash
# Basic parsing
python parse_segy.py survey.sgy

# Custom output directory
python parse_segy.py survey.sgy --output-dir ./results

# Fast processing (10-15s)
python parse_segy.py survey.sgy --config fast

# Accurate analysis with verbose output
python parse_segy.py survey.sgy --config accurate --verbose
```

**Configuration Options:**
- `fast` - Quick processing with minimal validation
- `balanced` - Good balance of speed and accuracy (default)
- `accurate` - Thorough analysis with comprehensive validation

## 🎯 Main Function Features

### **Interactive Menu**
```
🚀 AI-POWERED SEGY METADATA PARSER
   AI-Powered Seismic Data Analysis
================================================================================

🎯 What would you like to do?

1. 📋 Parse SEGY File
   Extract attributes, geometry, and revision information

2. 👁️  View SEGY Header  
   Display textual header content for analysis

3. ❓ Help & Documentation
   View usage examples and configuration options
```

### **Smart Defaults**
- **Auto-detects** SEGY files from `segypaths.txt`
- **Batch processing** - processes ALL files in `segypaths.txt` automatically
- **Recommends** balanced configuration (15-30s processing)
- **Creates** `./output` directory automatically
- **Handles** all file path and configuration complexity

### **Direct Commands** (Skip Menu)
```bash
# Parse directly
python main.py parse survey.sgy

# View header directly  
python main.py view survey.sgy

# Show help
python main.py help
```

### **Header Viewing**
```bash
# View SEGY textual header
python print_textual_header.py survey.sgy

# View header from segypaths.txt
python print_textual_header.py
```

## 📋 Essential Setup Files

### **`.env` - Complete LLM Configuration**
Create a `.env` file in the project root with your LLM settings:

**Setup .env File:**
```bash
# Copy the template and configure your API key
cp .env.template .env

# Edit .env with your API key
# Get your free API key from https://aistudio.google.com/
```

**Complete .env Template:** (see `.env.template` file)
```bash
# LLM Provider Configuration Template
DEFAULT_LLM_PROVIDER=gemini
FALLBACK_LLM_PROVIDERS=gemini

# Gemini Configuration (Recommended - Free)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TIMEOUT=30
GEMINI_MAX_RETRIES=1

# Optional: Local LLM Configuration
# LOCAL_LLM_SERVER_URL=http://your-server/v1/chat/completions
# LOCAL_LLM_MODEL=your_model_name
```

**Quick Setup for Gemini (Most Users):**
1. Get free API key from https://aistudio.google.com/
2. Replace `your_gemini_api_key_here` with your actual key
3. That's it! The defaults work for most users.

### **`segypaths.txt` - SEGY File Paths** (Optional)
```
# Add your SEGY file paths (one per line)
# All files will be processed in batch mode!
/path/to/survey1.sgy
./data/survey2.sgy
C:\seismic\survey3.sgy
```

**Batch Processing:** If you put multiple paths in `segypaths.txt`, the system will process ALL of them automatically! Perfect for analyzing multiple surveys at once.

## 🎛️ Processing Modes

The main function asks you to choose:

| Mode | Time | Use Case |
|------|------|----------|
| **Fast** | 10-15s | Quick exploration, large files |
| **Balanced** | 15-30s | **Recommended** - best balance |
| **Accurate** | 30-60s | Research, critical analysis |

## 📊 What You Get

### **Instant Results**
```
✅ Processing completed successfully!
⏱️  Total time: 12.46 seconds
📊 SEGY revision: None
🎯 Attributes found: 6
✅ Attributes validated: 6/6
🗺️  Geometric coordinates: 2

🎯 Top Attribute Mappings:
   1. ✅ CDP NUMBER: bytes 21-24 (confidence: 0.900)
   2. ✅ EASTING OF CDP: bytes 193-196 (confidence: 0.900)
   3. ✅ NORTHING OF CDP: bytes 197-200 (confidence: 0.900)
```

### **Output Files** (in `./output/`)
- `*_parsing_results.txt` - Human-readable report
- `*_parsing_results.json` - Machine-readable data
- `*_parsing_summary.csv` - Spreadsheet format

## 🆘 Common Issues

**"No LLM provider available"**
→ Check your `.env` file has `GEMINI_API_KEY=your_key`

**"File not found"**  
→ Either specify full path or add to `segypaths.txt`

**Want faster processing?**
→ Choose "Fast" mode when prompted

**Need more detail?**
→ Choose "Accurate" mode for research-grade analysis

## 🔧 Advanced: Customizing the Ontology

Want to improve AI accuracy for your specific data? You can customize the ontology (attribute definitions) that provides context to the LLM:

**Edit the ontology in `core/utils.py`:**
```python
# Add your domain-specific attributes
SEGY_ATTRIBUTE_ONTOLOGY = {
    "YOUR_CUSTOM_ATTRIBUTE": {
        "description": "Your specific attribute description",
        "typical_bytes": [25, 28],  # Common byte locations
        "aliases": ["ALT_NAME", "VARIANT_NAME"]
    }
}
```

**Why customize the ontology?**
- **Better accuracy** for your specific SEGY format conventions
- **Domain-specific attributes** that aren't in the standard ontology
- **Custom naming conventions** used by your organization
- **Improved LLM context** for specialized seismic data types

The ontology acts as a knowledge base that guides the AI's understanding of your SEGY files, leading to more accurate attribute detection and mapping.

## 🎉 That's It!

The main function handles all the complexity. Just run `python main.py` and follow the prompts!