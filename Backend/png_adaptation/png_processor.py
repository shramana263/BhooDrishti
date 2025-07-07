"""
PNG Image Processor for Change Detection Demo
Converts PNG satellite images to format compatible with existing system
"""

import numpy as np
# import cv2
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Import existing system components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class PNGSatelliteProcessor:
    """
    Process PNG satellite images for change detection using existing system architecture
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize PNG processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Simulated band mappings for RGB to multispectral conversion
        self.band_mappings = {
            'red': 0,      # R channel
            'green': 1,    # G channel  
            'blue': 2,     # B channel
            'nir': None,   # Will be simulated
            'swir1': None, # Will be simulated
            'swir2': None  # Will be simulated
        }
        
    def load_png_image(self, image_path: str) -> Dict:
        """
        Load PNG satellite image and convert to analysis-ready format
        
        Args:
            image_path: Path to PNG file
            
        Returns:
            Dict: Image data and metadata compatible with existing system
        """
        try:
            print(f"📸 Loading PNG image: {os.path.basename(image_path)}")
            
            # Load image using PIL for better format support
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Normalize to 0-1 range
            if image_array.dtype == np.uint8:
                image_array = image_array.astype(np.float32) / 255.0
            
            # Transpose to (bands, height, width) format expected by system
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Simulate additional spectral bands
            simulated_bands = self._simulate_spectral_bands(image_array)
            
            # Combine RGB with simulated bands
            full_image = np.concatenate([image_array, simulated_bands], axis=0)
            
            # Create metadata compatible with existing system
            metadata = {
                'file_path': image_path,
                'width': full_image.shape[2],
                'height': full_image.shape[1],
                'count': full_image.shape[0],
                'dtype': 'float32',
                'crs': 'EPSG:4326',  # Assumed WGS84
                'transform': self._create_dummy_transform(full_image.shape[2], full_image.shape[1]),
                'bounds': self._create_dummy_bounds(),
                'resolution': (10.0, 10.0),  # Assumed 10m resolution
                'band_names': ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
            }
            
            print(f"✅ PNG loaded successfully:")
            print(f"   📐 Size: {metadata['width']} x {metadata['height']}")
            print(f"   🔢 Bands: {metadata['count']} (3 real + 3 simulated)")
            print(f"   📏 Assumed resolution: 10m")
            
            return {
                'data': full_image,
                'metadata': metadata,
                'original_rgb': image_array  # Keep original RGB for visualization
            }
            
        except Exception as e:
            print(f"❌ Failed to load PNG {image_path}: {e}")
            self.logger.error(f"Error loading PNG: {e}")
            return None
    
    def _simulate_spectral_bands(self, rgb_bands: np.ndarray) -> np.ndarray:
        """
        Simulate NIR and SWIR bands from RGB channels
        Uses empirical relationships between visible and infrared bands
        
        Args:
            rgb_bands: RGB bands array (3, height, width)
            
        Returns:
            np.ndarray: Simulated bands (3, height, width) for NIR, SWIR1, SWIR2
        """
        red = rgb_bands[0]
        green = rgb_bands[1] 
        blue = rgb_bands[2]
        
        # Simulate NIR band
        # Vegetation appears bright in NIR, water appears dark
        # Use green channel as base and enhance vegetation areas
        vegetation_mask = (green > red) & (green > blue)  # Simple vegetation detection
        water_mask = (blue > red) & (blue > green)  # Simple water detection
        
        nir_simulated = green.copy()
        nir_simulated[vegetation_mask] = np.minimum(green[vegetation_mask] * 1.5, 1.0)  # Enhance vegetation
        nir_simulated[water_mask] = blue[water_mask] * 0.3  # Darken water
        
        # Simulate SWIR1 band (1.6μm)
        # Sensitive to moisture content, vegetation structure
        swir1_simulated = (red * 0.4 + green * 0.4 + blue * 0.2)
        swir1_simulated = np.clip(swir1_simulated, 0, 1)
        
        # Simulate SWIR2 band (2.2μm) 
        # Sensitive to mineral content, dry vegetation
        swir2_simulated = (red * 0.5 + green * 0.3 + blue * 0.2)
        swir2_simulated = np.clip(swir2_simulated, 0, 1)
        
        # Add some noise to make it more realistic
        noise_level = 0.02
        nir_simulated += np.random.normal(0, noise_level, nir_simulated.shape)
        swir1_simulated += np.random.normal(0, noise_level, swir1_simulated.shape)
        swir2_simulated += np.random.normal(0, noise_level, swir2_simulated.shape)
        
        # Clip to valid range
        nir_simulated = np.clip(nir_simulated, 0, 1)
        swir1_simulated = np.clip(swir1_simulated, 0, 1)
        swir2_simulated = np.clip(swir2_simulated, 0, 1)
        
        return np.stack([nir_simulated, swir1_simulated, swir2_simulated], axis=0)
    
    def _create_dummy_transform(self, width: int, height: int):
        """Create dummy geospatial transform for the image"""
        # Simple transform for demo - places image around Bengaluru coordinates
        # In real application, this would come from geospatial metadata
        from rasterio.transform import from_bounds
        
        # Approximate bounds around Bengaluru area
        west, south, east, north = 77.4, 12.8, 77.8, 13.2
        
        return from_bounds(west, south, east, north, width, height)
    
    def _create_dummy_bounds(self) -> Tuple[float, float, float, float]:
        """Create dummy bounds for the image"""
        # Approximate bounds around Bengaluru area  
        return (77.4, 12.8, 77.8, 13.2)  # (west, south, east, north)
    
    def calculate_spectral_indices(self, image_data: np.ndarray, band_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate spectral indices compatible with existing system
        
        Args:
            image_data: Image data array (bands, height, width)
            band_names: List of band names
            
        Returns:
            Dict: Dictionary of calculated indices
        """
        indices = {}
        band_dict = {name: i for i, name in enumerate(band_names)}
        
        try:
            # NDVI (Normalized Difference Vegetation Index)
            if 'nir' in band_dict and 'red' in band_dict:
                nir = image_data[band_dict['nir']]
                red = image_data[band_dict['red']]
                ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
                indices['ndvi'] = np.clip(ndvi, -1, 1)
                print("✅ Calculated NDVI")
            
            # NDBI (Normalized Difference Built-up Index)
            if 'swir1' in band_dict and 'nir' in band_dict:
                swir1 = image_data[band_dict['swir1']]
                nir = image_data[band_dict['nir']]
                ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
                indices['ndbi'] = np.clip(ndbi, -1, 1)
                print("✅ Calculated NDBI")
            
            # MNDWI (Modified Normalized Difference Water Index)
            if 'green' in band_dict and 'swir1' in band_dict:
                green = image_data[band_dict['green']]
                swir1 = image_data[band_dict['swir1']]
                mndwi = (green - swir1) / (green + swir1 + 1e-8)
                indices['mndwi'] = np.clip(mndwi, -1, 1)
                print("✅ Calculated MNDWI")
            
            # EVI (Enhanced Vegetation Index)
            if all(band in band_dict for band in ['nir', 'red', 'blue']):
                nir = image_data[band_dict['nir']]
                red = image_data[band_dict['red']]
                blue = image_data[band_dict['blue']]
                evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
                indices['evi'] = np.clip(evi, -1, 1)
                print("✅ Calculated EVI")
            
            # SAVI (Soil Adjusted Vegetation Index)
            if 'nir' in band_dict and 'red' in band_dict:
                nir = image_data[band_dict['nir']]
                red = image_data[band_dict['red']]
                L = 0.5  # Soil brightness correction factor
                savi = ((nir - red) / (nir + red + L)) * (1 + L)
                indices['savi'] = np.clip(savi, -1, 1)
                print("✅ Calculated SAVI")
            
            # NBR (Normalized Burn Ratio)
            if 'nir' in band_dict and 'swir2' in band_dict:
                nir = image_data[band_dict['nir']]
                swir2 = image_data[band_dict['swir2']]
                nbr = (nir - swir2) / (nir + swir2 + 1e-8)
                indices['nbr'] = np.clip(nbr, -1, 1)
                print("✅ Calculated NBR")
            
            print(f"📊 Calculated {len(indices)} spectral indices")
            
        except Exception as e:
            print(f"⚠️  Error calculating indices: {e}")
            self.logger.error(f"Error calculating spectral indices: {e}")
        
        return indices
    
    def create_mock_ee_image(self, image_data: np.ndarray, metadata: Dict, indices: Dict = None) -> 'MockEEImage':
        """
        Create a mock Earth Engine Image object for compatibility with existing system
        
        Args:
            image_data: Image data array
            metadata: Image metadata
            indices: Calculated spectral indices
            
        Returns:
            MockEEImage: Mock EE Image object
        """
        return MockEEImage(image_data, metadata, indices)
    
    def visualize_png_analysis(self, image1_path: str, image2_path: str, 
                              save_path: Optional[str] = None) -> None:
        """
        Create visualization comparing two PNG images
        
        Args:
            image1_path: Path to first PNG image
            image2_path: Path to second PNG image
            save_path: Optional path to save visualization
        """
        # Load both images
        img1 = self.load_png_image(image1_path)
        img2 = self.load_png_image(image2_path)
        
        if not img1 or not img2:
            print("❌ Failed to load images for visualization")
            return
        
        # Create RGB composites for display
        rgb1 = np.transpose(img1['original_rgb'], (1, 2, 0))
        rgb2 = np.transpose(img2['original_rgb'], (1, 2, 0))
        
        # Calculate indices
        indices1 = self.calculate_spectral_indices(img1['data'], img1['metadata']['band_names'])
        indices2 = self.calculate_spectral_indices(img2['data'], img2['metadata']['band_names'])
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # RGB images
        axes[0, 0].imshow(rgb1)
        axes[0, 0].set_title(f'Image 1: {os.path.basename(image1_path)}')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(rgb2)
        axes[1, 0].set_title(f'Image 2: {os.path.basename(image2_path)}')
        axes[1, 0].axis('off')
        
        # NDVI comparison
        if 'ndvi' in indices1 and 'ndvi' in indices2:
            im1 = axes[0, 1].imshow(indices1['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
            axes[0, 1].set_title('NDVI - Image 1')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
            
            im2 = axes[1, 1].imshow(indices2['ndvi'], cmap='RdYlGn', vmin=-1, vmax=1)
            axes[1, 1].set_title('NDVI - Image 2')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        # NDVI Difference
        if 'ndvi' in indices1 and 'ndvi' in indices2:
            ndvi_diff = indices2['ndvi'] - indices1['ndvi']
            im_diff = axes[0, 2].imshow(ndvi_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[0, 2].set_title('NDVI Change (Image2 - Image1)')
            axes[0, 2].axis('off')
            plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046)
            
            # Change detection mask
            change_mask = np.abs(ndvi_diff) > 0.1  # Threshold for significant change
            axes[1, 2].imshow(change_mask, cmap='Reds')
            axes[1, 2].set_title('Significant NDVI Changes')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: {save_path}")
        
        # Don't show plot in non-interactive environment
        if matplotlib.get_backend() != 'Agg':
            plt.show()
        
        plt.close()  # Close figure to free memory
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'processing': {
                'enable_noise_reduction': True,
                'enable_enhancement': True,
                'output_format': 'numpy'
            },
            'simulation': {
                'nir_enhancement_factor': 1.5,
                'noise_level': 0.02,
                'enable_band_simulation': True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('PNGSatelliteProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


class MockEEImage:
    """
    Mock Earth Engine Image class for compatibility with existing change detection system
    """
    
    def __init__(self, image_data: np.ndarray, metadata: Dict, indices: Dict = None):
        self.data = image_data
        self.metadata = metadata
        self.indices = indices or {}
        self.band_names = metadata['band_names']
        self._properties = {}
    
    def select(self, bands: Union[str, List[str]]) -> 'MockEEImage':
        """Mock select method"""
        if isinstance(bands, str):
            bands = [bands]
        
        # Get indices of selected bands
        band_indices = []
        selected_data = []
        
        for band in bands:
            if band in self.band_names:
                idx = self.band_names.index(band)
                band_indices.append(idx)
                selected_data.append(self.data[idx])
            elif band in self.indices:
                selected_data.append(self.indices[band])
        
        if selected_data:
            new_data = np.stack(selected_data, axis=0)
            new_metadata = self.metadata.copy()
            new_metadata['band_names'] = bands
            return MockEEImage(new_data, new_metadata, self.indices)
        
        return self
    
    def normalizedDifference(self, bands: List[str]) -> 'MockEEImage':
        """Mock normalized difference calculation"""
        if len(bands) != 2:
            raise ValueError("normalizedDifference requires exactly 2 bands")
        
        band1_data = self._get_band_data(bands[0])
        band2_data = self._get_band_data(bands[1])
        
        # Calculate normalized difference
        diff = (band1_data - band2_data) / (band1_data + band2_data + 1e-8)
        diff = np.clip(diff, -1, 1)
        
        # Create new image with difference
        new_metadata = self.metadata.copy()
        new_metadata['band_names'] = [f'{bands[0]}_{bands[1]}_diff']
        
        return MockEEImage(diff[np.newaxis, :, :], new_metadata)
    
    def subtract(self, other: 'MockEEImage') -> 'MockEEImage':
        """Mock subtract operation"""
        diff_data = self.data - other.data
        new_metadata = self.metadata.copy()
        new_metadata['band_names'] = [f'{name}_diff' for name in self.band_names]
        
        return MockEEImage(diff_data, new_metadata)
    
    def lt(self, threshold: float) -> 'MockEEImage':
        """Mock less than operation"""
        mask_data = (self.data < threshold).astype(np.float32)
        new_metadata = self.metadata.copy()
        new_metadata['band_names'] = [f'{name}_lt_{threshold}' for name in self.band_names]
        
        return MockEEImage(mask_data, new_metadata)
    
    def gt(self, threshold: float) -> 'MockEEImage':
        """Mock greater than operation"""
        mask_data = (self.data > threshold).astype(np.float32)
        new_metadata = self.metadata.copy()
        new_metadata['band_names'] = [f'{name}_gt_{threshold}' for name in self.band_names]
        
        return MockEEImage(mask_data, new_metadata)
    
    def And(self, other: 'MockEEImage') -> 'MockEEImage':
        """Mock AND operation"""
        and_data = np.logical_and(self.data > 0, other.data > 0).astype(np.float32)
        new_metadata = self.metadata.copy()
        
        return MockEEImage(and_data, new_metadata)
    
    def bandNames(self) -> 'MockEEList':
        """Mock bandNames method"""
        return MockEEList(self.band_names)
    
    def multiply(self, value) -> 'MockEEImage':
        """Mock multiply operation"""
        if hasattr(value, 'data'):  # Another MockEEImage
            mult_data = self.data * value.data
        else:  # Scalar or array
            mult_data = self.data * value
        
        return MockEEImage(mult_data, self.metadata)
    
    def _get_band_data(self, band_name: str) -> np.ndarray:
        """Get data for a specific band"""
        if band_name in self.band_names:
            idx = self.band_names.index(band_name)
            return self.data[idx]
        elif band_name in self.indices:
            return self.indices[band_name]
        else:
            raise ValueError(f"Band {band_name} not found")
    
    def rename(self, name: str) -> 'MockEEImage':
        """Mock rename operation"""
        new_metadata = self.metadata.copy()
        new_metadata['band_names'] = [name]
        return MockEEImage(self.data, new_metadata, self.indices)
    
    def getInfo(self) -> Dict:
        """Mock getInfo method"""
        return {
            'bands': [{'id': name} for name in self.band_names],
            'properties': self._properties
        }


class MockEEList:
    """Mock Earth Engine List class"""
    
    def __init__(self, items: List):
        self.items = items
    
    def getInfo(self) -> List:
        """Mock getInfo method"""
        return self.items


class MockEEGeometry:
    """Mock Earth Engine Geometry class"""
    
    def __init__(self, bounds: Tuple[float, float, float, float]):
        self.bounds = bounds  # (west, south, east, north)
    
    def area(self) -> 'MockEENumber':
        """Mock area calculation"""
        west, south, east, north = self.bounds
        # Rough area calculation in square meters
        width_deg = east - west
        height_deg = north - south
        
        # Convert to approximate meters (very rough calculation)
        meters_per_degree = 111320  # At equator
        area_m2 = width_deg * height_deg * (meters_per_degree ** 2)
        
        return MockEENumber(area_m2)


class MockEENumber:
    """Mock Earth Engine Number class"""
    
    def __init__(self, value: float):
        self.value = value
    
    def getInfo(self) -> float:
        """Mock getInfo method"""
        return self.value


def demo_png_processing():
    """
    Demonstrate PNG processing workflow
    """
    print("🔧 PNG PROCESSING DEMO")
    print("=" * 50)
    
    # Initialize processor
    processor = PNGSatelliteProcessor()
    
    # Define image paths
    image1_path = "/home/parambrata-ghosh/Development/Personal/Hackathon/ISRO/BhooDristi/Backend/data/raw/kpc_2014.png"
    image2_path = "/home/parambrata-ghosh/Development/Personal/Hackathon/ISRO/BhooDristi/Backend/data/raw/kpc_2022.png"
    
    # Check if files exist
    if not os.path.exists(image1_path):
        print(f"❌ Image 1 not found: {image1_path}")
        return
    
    if not os.path.exists(image2_path):
        print(f"❌ Image 2 not found: {image2_path}")
        return
    
    # Process images
    print("\n1. Loading and processing PNG images...")
    img1 = processor.load_png_image(image1_path)
    img2 = processor.load_png_image(image2_path)
    
    if not img1 or not img2:
        print("❌ Failed to load images")
        return
    
    # Calculate spectral indices
    print("\n2. Calculating spectral indices...")
    indices1 = processor.calculate_spectral_indices(img1['data'], img1['metadata']['band_names'])
    indices2 = processor.calculate_spectral_indices(img2['data'], img2['metadata']['band_names'])
    
    # Create visualization
    print("\n3. Creating comparison visualization...")
    output_dir = Path("/home/parambrata-ghosh/Development/Personal/Hackathon/ISRO/BhooDristi/Backend/png_adaptation/outputs")
    output_dir.mkdir(exist_ok=True)
    
    viz_path = output_dir / "png_comparison.png"
    processor.visualize_png_analysis(image1_path, image2_path, str(viz_path))
    
    print("\n✅ PNG processing demo completed!")
    print(f"📊 Results saved to: {output_dir}")


if __name__ == "__main__":
    demo_png_processing()
