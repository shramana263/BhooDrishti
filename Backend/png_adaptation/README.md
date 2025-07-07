# PNG Adaptation for BhooDrishti Change Detection

This folder contains the PNG adaptation module that allows the BhooDrishti change detection system to work with PNG satellite images instead of requiring Google Earth Engine.

## Overview

The PNG adaptation maintains compatibility with the existing system architecture while enabling change detection analysis using local PNG satellite images. This is particularly useful for demonstrations, offline analysis, or when GEE access is not available.

## Features

- **PNG Image Processing**: Load and process PNG satellite images
- **Spectral Band Simulation**: Simulate NIR and SWIR bands from RGB channels
- **Change Detection**: Detect vegetation, urban, and water body changes
- **Analysis Engine**: Comprehensive analysis with risk assessment and recommendations
- **Visualization**: Create detailed visualizations and dashboard
- **Compatibility**: Uses same interfaces as the existing system

## Architecture

```
png_adaptation/
├── __init__.py                 # Module initialization
├── png_processor.py           # PNG image processing and band simulation
├── png_change_detection.py    # Change detection adapted for PNG
├── png_change_analysis.py     # Change analysis and reporting
├── demo_runner.py            # Complete demonstration script
├── requirements.txt          # Dependencies
├── README.md                # This file
└── outputs/                 # Generated outputs (created when run)
```

## Key Components

### 1. PNG Processor (`png_processor.py`)
- Loads PNG satellite images
- Simulates NIR and SWIR bands from RGB channels
- Calculates spectral indices (NDVI, NDBI, MNDWI, etc.)
- Creates mock Earth Engine objects for compatibility

### 2. Change Detection Engine (`png_change_detection.py`)
- Adapts existing change detection algorithms for PNG images
- Detects vegetation changes using NDVI analysis
- Identifies urban expansion patterns
- Monitors water body changes
- Provides comprehensive change detection workflow

### 3. Analysis Engine (`png_change_analysis.py`)
- Detailed analysis of detected changes
- Risk assessment and severity classification
- Generates recommendations
- Creates comprehensive reports
- Exports results in JSON format

### 4. Demo Runner (`demo_runner.py`)
- Complete demonstration workflow
- Processes the included satellite images
- Generates all visualizations and reports
- Creates a comprehensive dashboard

## Usage

### Quick Start

1. **Run the complete demo:**
```bash
cd png_adaptation
python demo_runner.py
```

This will process the satellite images and generate all outputs in the `outputs/` folder.

### Custom Analysis

```python
from png_adaptation import PNGSatelliteProcessor, PNGChangeDetectionEngine, PNGChangeAnalysisEngine

# Initialize components
processor = PNGSatelliteProcessor()
detector = PNGChangeDetectionEngine()
analyzer = PNGChangeAnalysisEngine()

# Process images
img1_data = processor.load_png_image("path/to/image1.png")
img2_data = processor.load_png_image("path/to/image2.png")

# Detect changes
changes = detector.comprehensive_change_detection_png(
    "path/to/image1.png", 
    "path/to/image2.png"
)

# Generate analysis
report = analyzer.generate_comprehensive_report_png(
    changes, "path/to/image1.png", "path/to/image2.png"
)
```

## Input Data

The system expects PNG satellite images. The demo uses:
- `kpc_2014.png` - Karnataka state satellite image from 2014
- `kpc_2022.png` - Karnataka state satellite image from 2022

These images should be located in `../data/raw/` relative to this folder.

## Outputs

When you run the demo, the following outputs are generated in the `outputs/` folder:

### Visualizations
- `change_detection_dashboard.png` - Comprehensive dashboard
- `comprehensive_change_detection.png` - Multi-panel change analysis
- `rgb_comparison.png` - Side-by-side image comparison
- `ndvi_change.png` - NDVI change map
- `urban_expansion.png` - Urban expansion areas

### Reports
- `comprehensive_analysis_report.json` - Detailed analysis in JSON format

## Technical Details

### Band Simulation
Since PNG images only contain RGB channels, the system simulates additional spectral bands:

- **NIR (Near-Infrared)**: Simulated by enhancing vegetation areas in the green channel
- **SWIR1 (Short-Wave Infrared 1)**: Simulated using weighted combination of RGB
- **SWIR2 (Short-Wave Infrared 2)**: Simulated for mineral and dry vegetation detection

### Spectral Indices
The following indices are calculated from the simulated bands:
- **NDVI**: Vegetation health and coverage
- **NDBI**: Built-up area detection
- **MNDWI**: Water body identification
- **EVI**: Enhanced vegetation index
- **SAVI**: Soil-adjusted vegetation index
- **NBR**: Normalized burn ratio

### Change Detection Methods
- **Vegetation Changes**: NDVI differencing with configurable thresholds
- **Urban Expansion**: NDBI analysis and brightness change detection
- **Water Changes**: MNDWI-based water mask comparison

## Configuration

The system accepts configuration dictionaries to customize:

```python
config = {
    'change_detection': {
        'ndvi_threshold': 0.1,      # NDVI change threshold
        'urban_threshold': 0.05,     # Urban expansion threshold
        'water_threshold': 0.3,      # Water change threshold
        'confidence_threshold': 0.5   # Confidence threshold
    },
    'analysis': {
        'significant_change_area': 1000,  # Minimum significant area (m²)
        'alert_thresholds': {
            'deforestation': 5000,        # Deforestation alert (m²)
            'urban_expansion': 2000,      # Urban expansion alert (m²)
            'water_loss': 3000           # Water loss alert (m²)
        }
    }
}
```

## Limitations

1. **Spectral Bands**: NIR and SWIR bands are simulated, not real
2. **Atmospheric Correction**: Not applied to PNG images
3. **Geometric Accuracy**: Depends on input image quality
4. **Temporal Resolution**: Limited to available PNG images
5. **Spatial Resolution**: Dependent on input image resolution

## Integration with Main System

This PNG adaptation is designed to be completely compatible with the existing BhooDrishti system:

- Uses the same configuration structure
- Provides similar output formats
- Maintains the same workflow patterns
- Can be easily swapped with the GEE-based system

## Dependencies

See `requirements.txt` for a complete list. Main dependencies:
- numpy, matplotlib (visualization)
- PIL/Pillow (image processing)
- opencv-python (image operations)
- scipy (scientific computing)
- rasterio (geospatial operations)

## Example Results

The demo analysis of Karnataka state (2014-2022) typically shows:
- **Vegetation Changes**: Forest loss and agricultural changes
- **Urban Expansion**: Growth around major cities
- **Water Body Changes**: Seasonal and permanent water changes

Results include:
- Change area calculations in hectares
- Severity assessments
- Risk level classifications
- Actionable recommendations
- Visual change maps and dashboard

## Support

This adaptation module provides a complete alternative to Google Earth Engine for change detection analysis, making the BhooDrishti system accessible for offline demonstrations and analysis scenarios.
