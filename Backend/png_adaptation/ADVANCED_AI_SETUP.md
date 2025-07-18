# Advanced Deep Learning Setup Guide for BhooDrishti
# =====================================================

This guide explains how to set up the advanced deep learning features for BhooDrishti change detection system.

## Overview

The advanced AI features provide state-of-the-art change detection capabilities using:
- **Siamese U-Net with Attention**: Advanced neural architecture for precise change detection
- **Cloud Detection Network**: Automated cloud and shadow masking
- **Quality Assessment**: Image quality evaluation for confidence scoring
- **Multi-scale Analysis**: Comprehensive change type classification

## System Requirements

### Minimum Requirements (CPU-only)
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Python**: 3.8 or higher
- **OS**: Windows 10/11, Linux, macOS

### Recommended Requirements (GPU-accelerated)
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 6+ GB VRAM
- **Storage**: 10 GB free space
- **CUDA**: 11.8 or compatible
- **Python**: 3.9 or higher

## Installation Steps

### Step 1: Install Basic Requirements
```bash
# Navigate to backend directory
cd Backend

# Install basic requirements
pip install -r requirements.txt
```

### Step 2: Install Advanced AI Requirements
```bash
# Install advanced deep learning dependencies
pip install -r png_adaptation/requirements_advanced.txt
```

### Step 3: GPU Support (Recommended)
For NVIDIA GPUs with CUDA support:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For systems without CUDA:
```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Verify Installation
```bash
# Test the advanced processor
cd png_adaptation
python advanced_dl_processor.py
```

## Key Dependencies

### Core Deep Learning
- **torch**: >=2.0.0 (PyTorch framework)
- **torchvision**: >=0.15.0 (Computer vision utilities)
- **albumentations**: >=1.3.0 (Advanced augmentations)
- **segmentation-models-pytorch**: >=0.3.3 (Pre-trained models)

### Computer Vision
- **opencv-python**: >=4.8.0 (Image processing)
- **scikit-image**: >=0.21.0 (Image analysis)
- **Pillow**: >=10.0.0 (Image handling)

### Scientific Computing
- **numpy**: >=1.24.0 (Numerical computing)
- **scipy**: >=1.10.0 (Scientific algorithms)
- **scikit-learn**: >=1.3.0 (Machine learning utilities)

### Optional but Recommended
- **wandb**: For experiment tracking
- **tensorboard**: For visualization
- **xformers**: For memory-efficient attention

## API Endpoints

### Advanced AI Analysis
```http
POST /predict/advanced
Content-Type: multipart/form-data

Parameters:
- image1: Earlier satellite image
- image2: Later satellite image
- config: JSON configuration (optional)
- model_type: "siamese_unet" (default)
```

### Model Information
```http
GET /models/info
```

### System Requirements
```http
GET /system/requirements
```

## Usage Examples

### Basic Advanced Analysis
```python
import requests

# Prepare files
files = {
    'image1': open('image_2014.png', 'rb'),
    'image2': open('image_2022.png', 'rb')
}

# Submit for advanced AI analysis
response = requests.post(
    'http://localhost:8000/predict/advanced',
    files=files
)

result = response.json()
print(f"Analysis ID: {result['analysis_id']}")
print(f"Device used: {result['model_info']['device']}")
```

### Custom Configuration
```python
import json

config = {
    "model": {
        "image_size": 512,  # Higher resolution
        "confidence_threshold": 0.7
    },
    "analysis": {
        "pixel_area_m2": 10.0,  # 10m² per pixel
        "min_change_area_ha": 0.5
    }
}

files = {
    'image1': open('image_2014.png', 'rb'),
    'image2': open('image_2022.png', 'rb')
}

data = {
    'config': json.dumps(config),
    'model_type': 'siamese_unet'
}

response = requests.post(
    'http://localhost:8000/predict/advanced',
    files=files,
    data=data
)
```

## Performance Optimization

### GPU Memory Management
- Default image size: 256px (adjust based on GPU memory)
- Batch processing: Currently single image pairs
- Memory cleanup: Automatic after processing

### CPU-only Performance
- Reduce image size to 128-256px
- Use lightweight models
- Consider parallel processing for multiple analyses

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce image size in config
   {"model": {"image_size": 128}}
   ```

2. **Import Errors**
   ```bash
   # Reinstall requirements
   pip install --force-reinstall -r png_adaptation/requirements_advanced.txt
   ```

3. **Slow Performance**
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Error Messages
- "Advanced AI features not available": Install advanced requirements
- "CUDA out of memory": Reduce image size or use CPU
- "Model loading failed": Check PyTorch installation

## Model Architecture Details

### Siamese U-Net Architecture
```
Input Images (2x 3x256x256)
     ↓
Shared Encoder (ResNet-like)
     ↓
Spatial Attention Module
     ↓
Feature Fusion Layer
     ↓
U-Net Decoder
     ↓
Change Map Output (1x256x256)
```

### Additional Networks
- **Cloud Detector**: 3-class segmentation (clear/cloud/shadow)
- **Quality Assessor**: Single quality score (0-1)
- **Post-processor**: Confidence-weighted final output

## Comparison with Basic Methods

| Feature | Basic PNG | Advanced AI |
|---------|-----------|-------------|
| Accuracy | Good | Very High |
| Speed | Fast (30s) | Moderate (2-5min) |
| Cloud Detection | Simple | Sophisticated |
| Quality Assessment | None | Comprehensive |
| Change Types | 3 basic | Detailed classification |
| Confidence Scoring | Basic | AI-powered |
| GPU Support | No | Yes |

## Future Enhancements

Planned improvements:
- Vision Transformer models
- Self-supervised learning
- Multi-temporal analysis
- Real-time processing
- Model fine-tuning interface

## Support

For technical support:
1. Check logs in the terminal
2. Verify GPU/CUDA installation
3. Review system requirements
4. Test with smaller images first

The advanced AI features significantly improve change detection accuracy while providing detailed analysis and confidence metrics.
