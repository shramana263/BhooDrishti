"""
Robust Preprocessing Module for Satellite Change Detection
Handles cloud/shadow masking, noise reduction, and quality assessment
"""

import ee
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging
from pathlib import Path
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

class SatellitePreprocessor:
    """
    Comprehensive preprocessing pipeline for satellite imagery
    Handles multiple sensors with robust cloud/shadow masking
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            'cloud_coverage': 0.15,  # 15% max cloud coverage
            'shadow_coverage': 0.10,  # 10% max shadow coverage
            'data_availability': 0.80,  # 80% min valid pixels
            'temporal_consistency': 0.85,  # 85% min temporal consistency
            'spectral_quality': 0.75   # 75% min spectral quality
        }
        
        # Sensor-specific configurations
        self.sensor_configs = {
            'sentinel2': {
                'cloud_prob_threshold': 60,
                'shadow_prob_threshold': 40,
                'qa_bands': ['QA60', 'SCL'],
                'scale': 10,
                'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
            },
            'landsat8': {
                'cloud_prob_threshold': 50,
                'shadow_prob_threshold': 30,
                'qa_bands': ['QA_PIXEL', 'QA_RADSAT'],
                'scale': 30,
                'bands': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            },
            'landsat9': {
                'cloud_prob_threshold': 50,
                'shadow_prob_threshold': 30,
                'qa_bands': ['QA_PIXEL', 'QA_RADSAT'],
                'scale': 30,
                'bands': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            }
        }
        
    def load_satellite_image(self, file_path: str) -> Dict:
        """
        Load satellite image and extract metadata.
        
        Args:
            file_path: Path to GeoTIFF file
            
        Returns:
            Dict: Image data and metadata
        """
        try:
            with rasterio.open(file_path) as src:
                # Read all bands
                data = src.read()
                
                # Get metadata
                metadata = {
                    'file_path': file_path,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'resolution': src.res,
                    'band_names': self._get_band_names(src.count)
                }
                
                print(f"✅ Loaded: {os.path.basename(file_path)}")
                print(f"   📐 Size: {metadata['width']} x {metadata['height']}")
                print(f"   🔢 Bands: {metadata['count']}")
                print(f"   📏 Resolution: {metadata['resolution'][0]:.1f}m")
                
                return {
                    'data': data,
                    'metadata': metadata
                }
                
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            return None
    
    def _get_band_names(self, band_count: int) -> List[str]:
        """
        Get standard band names based on band count.
        
        Args:
            band_count: Number of bands
            
        Returns:
            List[str]: Band names
        """
        # Standard band order for processed images
        standard_bands = [
            'red', 'green', 'blue', 'nir', 'swir1', 'swir2',
            'ndvi', 'ndwi', 'ndbi', 'evi'
        ]
        
        return standard_bands[:band_count]
    
    def normalize_reflectance_values(self, data: np.ndarray, 
                                   satellite: str = 'sentinel2') -> np.ndarray:
        """
        Normalize reflectance values to 0-1 range.
        
        Args:
            data: Input data array
            satellite: Satellite type for scaling
            
        Returns:
            np.ndarray: Normalized data
        """
        data_normalized = data.astype(np.float32)
        
        # Handle different satellites
        if satellite.lower() == 'sentinel2':
            # Sentinel-2 surface reflectance values are 0-10000
            data_normalized = np.clip(data_normalized / 10000.0, 0, 1)
        elif satellite.lower() in ['landsat8', 'landsat9']:
            # Landsat surface reflectance values are 0-65535
            data_normalized = np.clip(data_normalized / 65535.0, 0, 1)
        else:
            # Generic normalization - assume 16-bit data
            data_normalized = np.clip(data_normalized / 65535.0, 0, 1)
        
        return data_normalized
    
    def calculate_additional_indices(self, data: np.ndarray, 
                                   band_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate additional vegetation and land cover indices.
        
        Args:
            data: Input data array (bands, height, width)
            band_names: List of band names
            
        Returns:
            Dict: Dictionary of calculated indices
        """
        indices = {}
        
        # Get band indices
        band_indices = {name: i for i, name in enumerate(band_names)}
        
        try:
            # SAVI (Soil Adjusted Vegetation Index)
            if 'red' in band_indices and 'nir' in band_indices:
                red = data[band_indices['red']]
                nir = data[band_indices['nir']]
                L = 0.5  # Soil brightness correction factor
                savi = ((nir - red) / (nir + red + L)) * (1 + L)
                indices['savi'] = savi
            
            # GNDVI (Green Normalized Difference Vegetation Index)
            if 'green' in band_indices and 'nir' in band_indices:
                green = data[band_indices['green']]
                nir = data[band_indices['nir']]
                gndvi = (nir - green) / (nir + green)
                indices['gndvi'] = gndvi
            
            # NBR (Normalized Burn Ratio)
            if 'nir' in band_indices and 'swir2' in band_indices:
                nir = data[band_indices['nir']]
                swir2 = data[band_indices['swir2']]
                nbr = (nir - swir2) / (nir + swir2)
                indices['nbr'] = nbr
            
            # MNDWI (Modified Normalized Difference Water Index)
            if 'green' in band_indices and 'swir1' in band_indices:
                green = data[band_indices['green']]
                swir1 = data[band_indices['swir1']]
                mndwi = (green - swir1) / (green + swir1)
                indices['mndwi'] = mndwi
            
            print(f"📊 Calculated {len(indices)} additional indices")
            
        except Exception as e:
            print(f"⚠️  Error calculating indices: {e}")
        
        return indices
    
    def create_rgb_composite(self, data: np.ndarray, 
                           band_names: List[str],
                           enhance: bool = True) -> np.ndarray:
        """
        Create RGB composite for visualization.
        
        Args:
            data: Input data array
            band_names: List of band names
            enhance: Apply contrast enhancement
            
        Returns:
            np.ndarray: RGB composite
        """
        band_indices = {name: i for i, name in enumerate(band_names)}
        
        if all(band in band_indices for band in ['red', 'green', 'blue']):
            rgb = np.stack([
                data[band_indices['red']],
                data[band_indices['green']],
                data[band_indices['blue']]
            ], axis=0)
            
            if enhance:
                # Apply 2% linear stretch
                rgb = self._linear_stretch(rgb, 2, 98)
            
            # Transpose for matplotlib (H, W, C)
            rgb = np.transpose(rgb, (1, 2, 0))
            
            return rgb
        else:
            print("❌ RGB bands not found")
            return None
    
    def create_false_color_composite(self, data: np.ndarray, 
                                   band_names: List[str],
                                   composite_type: str = 'nir_red_green') -> np.ndarray:
        """
        Create false color composite.
        
        Args:
            data: Input data array
            band_names: List of band names
            composite_type: Type of false color composite
            
        Returns:
            np.ndarray: False color composite
        """
        band_indices = {name: i for i, name in enumerate(band_names)}
        
        composites = {
            'nir_red_green': ['nir', 'red', 'green'],  # Vegetation
            'swir_nir_red': ['swir1', 'nir', 'red'],   # Agriculture
            'swir_nir_green': ['swir1', 'nir', 'green'] # Urban
        }
        
        if composite_type not in composites:
            print(f"❌ Unknown composite type: {composite_type}")
            return None
        
        required_bands = composites[composite_type]
        
        if all(band in band_indices for band in required_bands):
            composite = np.stack([
                data[band_indices[required_bands[0]]],
                data[band_indices[required_bands[1]]],
                data[band_indices[required_bands[2]]]
            ], axis=0)
            
            # Apply enhancement
            composite = self._linear_stretch(composite, 2, 98)
            
            # Transpose for matplotlib
            composite = np.transpose(composite, (1, 2, 0))
            
            return composite
        else:
            print(f"❌ Required bands not found for {composite_type}")
            return None
    
    def _linear_stretch(self, data: np.ndarray, 
                       low_percentile: float = 2, 
                       high_percentile: float = 98) -> np.ndarray:
        """
        Apply linear contrast stretch.
        
        Args:
            data: Input data
            low_percentile: Lower percentile for stretch
            high_percentile: Upper percentile for stretch
            
        Returns:
            np.ndarray: Stretched data
        """
        stretched = np.zeros_like(data, dtype=np.float32)
        
        for i in range(data.shape[0]):
            band = data[i]
            valid_pixels = band[~np.isnan(band)]
            
            if len(valid_pixels) > 0:
                low_val = np.percentile(valid_pixels, low_percentile)
                high_val = np.percentile(valid_pixels, high_percentile)
                
                stretched[i] = np.clip((band - low_val) / (high_val - low_val), 0, 1)
        
        return stretched
    
    def prepare_for_change_detection(self, image1_path: str, 
                                   image2_path: str,
                                   output_dir: str = "data/processed") -> Dict:
        """
        Prepare two images for change detection analysis.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            output_dir: Output directory for processed files
            
        Returns:
            Dict: Processed data information
        """
        print("🔄 Preparing images for change detection...")
        
        # Load both images
        img1 = self.load_satellite_image(image1_path)
        img2 = self.load_satellite_image(image2_path)
        
        if not img1 or not img2:
            print("❌ Failed to load one or both images")
            return None
        
        # Check compatibility
        if img1['metadata']['crs'] != img2['metadata']['crs']:
            print("⚠️  Images have different CRS - consider reprojecting")
        
        if img1['metadata']['resolution'] != img2['metadata']['resolution']:
            print("⚠️  Images have different resolutions")
        
        # Normalize data
        img1_norm = self.normalize_reflectance_values(img1['data'], 'sentinel2')
        img2_norm = self.normalize_reflectance_values(img2['data'], 'sentinel2')
        
        # Calculate additional indices
        indices1 = self.calculate_additional_indices(img1_norm, img1['metadata']['band_names'])
        indices2 = self.calculate_additional_indices(img2_norm, img2['metadata']['band_names'])
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        processed_data = {
            'image1': {
                'data': img1_norm,
                'indices': indices1,
                'metadata': img1['metadata']
            },
            'image2': {
                'data': img2_norm,
                'indices': indices2,
                'metadata': img2['metadata']
            },
            'output_dir': str(output_path)
        }
        
        print("✅ Images prepared for change detection")
        return processed_data
    
    def visualize_comparison(self, image1_path: str, image2_path: str, 
                           save_path: Optional[str] = None):
        """
        Create side-by-side visualization of two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            save_path: Path to save visualization
        """
        # Load images
        img1 = self.load_satellite_image(image1_path)
        img2 = self.load_satellite_image(image2_path)
        
        if not img1 or not img2:
            print("❌ Failed to load images for visualization")
            return
        
        # Normalize data
        img1_norm = self.normalize_reflectance_values(img1['data'], 'sentinel2')
        img2_norm = self.normalize_reflectance_values(img2['data'], 'sentinel2')
        
        # Create RGB composites
        rgb1 = self.create_rgb_composite(img1_norm, img1['metadata']['band_names'])
        rgb2 = self.create_rgb_composite(img2_norm, img2['metadata']['band_names'])
        
        if rgb1 is None or rgb2 is None:
            print("❌ Failed to create RGB composites")
            return
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(rgb1)
        axes[0].set_title(f'Image 1\n{os.path.basename(image1_path)}')
        axes[0].axis('off')
        
        axes[1].imshow(rgb2)
        axes[1].set_title(f'Image 2\n{os.path.basename(image2_path)}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: {save_path}")
        
        plt.show()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            'preprocessing': {
                'enable_cloud_masking': True,
                'enable_shadow_masking': True,
                'enable_topographic_correction': True,
                'enable_atmospheric_correction': True,
                'cloud_buffer_distance': 50,  # meters
                'shadow_buffer_distance': 30,  # meters
                'temporal_window': 30,  # days
                'min_images_per_composite': 2
            },
            'quality_control': {
                'enable_outlier_detection': True,
                'enable_temporal_consistency_check': True,
                'enable_spectral_validation': True,
                'outlier_threshold': 2.5,  # standard deviations
                'temporal_threshold': 0.3   # correlation threshold
            },
            'output': {
                'save_quality_masks': True,
                'save_metadata': True,
                'compression': 'lzw'
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in user_config.items():
                        if key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('SatellitePreprocessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def advanced_cloud_masking(self, image: ee.Image, sensor: str) -> ee.Image:
        """
        Advanced cloud masking using multiple approaches
        
        Args:
            image: Earth Engine image
            sensor: Sensor type ('sentinel2', 'landsat8', 'landsat9')
            
        Returns:
            Masked image with cloud mask
        """
        sensor_config = self.sensor_configs[sensor]
        
        if sensor == 'sentinel2':
            return self._sentinel2_cloud_mask(image, sensor_config)
        elif sensor in ['landsat8', 'landsat9']:
            return self._landsat_cloud_mask(image, sensor_config)
        else:
            raise ValueError(f"Unsupported sensor: {sensor}")
    
    def _sentinel2_cloud_mask(self, image: ee.Image, config: Dict) -> ee.Image:
        """Advanced Sentinel-2 cloud masking"""
        # Method 1: QA60 band masking
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0)
        )
        
        # Method 2: SCL (Scene Classification) masking
        scl = image.select('SCL')
        # Clear sky (4, 5, 6, 7), snow/ice (11) are good
        scl_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        
        # Method 3: Spectral-based cloud detection
        # Blue band threshold for bright clouds
        blue_threshold = image.select('B2').lt(0.3)
        
        # NDSI for distinguishing clouds from snow
        ndsi = image.normalizedDifference(['B3', 'B11']).rename('NDSI')
        cloud_spectral = blue_threshold.And(ndsi.lt(0.4))
        
        # Combine all masks
        combined_mask = qa_mask.And(scl_mask).And(cloud_spectral.Not())
        
        # Apply buffer to cloud edges
        if self.config['preprocessing']['cloud_buffer_distance'] > 0:
            cloud_pixels = combined_mask.Not()
            buffered_clouds = cloud_pixels.focal_max(
                radius=self.config['preprocessing']['cloud_buffer_distance'],
                kernelType='circle',
                units='meters'
            )
            combined_mask = buffered_clouds.Not()
        
        return image.updateMask(combined_mask).set('cloud_mask', combined_mask)
    
    def _landsat_cloud_mask(self, image: ee.Image, config: Dict) -> ee.Image:
        """Advanced Landsat cloud masking"""
        # QA_PIXEL band masking
        qa = image.select('QA_PIXEL')
        
        # Bit positions for different quality flags
        cloud_bit = 1 << 3    # Cloud
        shadow_bit = 1 << 4   # Cloud shadow
        cirrus_bit = 1 << 2   # Cirrus
        
        # Create mask
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
            qa.bitwiseAnd(shadow_bit).eq(0)
        ).And(
            qa.bitwiseAnd(cirrus_bit).eq(0)
        )
        
        # Additional spectral-based cloud detection
        # Clouds are typically bright in visible bands
        blue = image.select('SR_B2')
        green = image.select('SR_B3')
        red = image.select('SR_B4')
        
        # Cloud probability based on brightness
        brightness = blue.add(green).add(red).divide(3)
        cloud_prob = brightness.gt(0.3)  # Threshold for bright pixels
        
        # Temperature-based cloud detection (clouds are cold)
        if 'ST_B10' in image.bandNames().getInfo():
            temp = image.select('ST_B10')
            cold_clouds = temp.lt(285)  # Kelvin, ~12°C
            mask = mask.And(cold_clouds.Not())
        
        # Final mask combining all methods
        final_mask = mask.And(cloud_prob.Not())
        
        return image.updateMask(final_mask).set('cloud_mask', final_mask)
    
    def shadow_masking(self, image: ee.Image, sensor: str) -> ee.Image:
        """
        Advanced shadow detection and masking
        
        Args:
            image: Earth Engine image
            sensor: Sensor type
            
        Returns:
            Image with shadow mask applied
        """
        # Method 1: Dark pixel detection
        if sensor == 'sentinel2':
            nir = image.select('B8')
            red = image.select('B4')
        else:  # Landsat
            nir = image.select('SR_B5')
            red = image.select('SR_B4')
        
        # Shadows are dark in NIR
        dark_pixels = nir.lt(0.15)
        
        # Method 2: NDVI-based shadow detection
        # Shadows have lower NDVI than surrounding vegetation
        ndvi = image.normalizedDifference([nir.bandNames().get(0), red.bandNames().get(0)])
        low_ndvi = ndvi.lt(0.3)
        
        # Method 3: Topographic shadow modeling (if DEM available)
        shadow_mask = dark_pixels.And(low_ndvi)
        
        # Apply shadow buffer
        if self.config['preprocessing']['shadow_buffer_distance'] > 0:
            shadow_pixels = shadow_mask
            buffered_shadows = shadow_pixels.focal_max(
                radius=self.config['preprocessing']['shadow_buffer_distance'],
                kernelType='circle',
                units='meters'
            )
            shadow_mask = buffered_shadows.Not()
        else:
            shadow_mask = shadow_mask.Not()
        
        return image.updateMask(shadow_mask).set('shadow_mask', shadow_mask)
    
    def atmospheric_correction(self, image: ee.Image, sensor: str) -> ee.Image:
        """
        Apply atmospheric correction using Dark Object Subtraction (DOS)
        
        Args:
            image: Earth Engine image
            sensor: Sensor type
            
        Returns:
            Atmospherically corrected image
        """
        if not self.config['preprocessing']['enable_atmospheric_correction']:
            return image
        
        # Get sensor configuration
        sensor_config = self.sensor_configs[sensor]
        bands = sensor_config['bands']
        
        # Dark Object Subtraction
        corrected_bands = []
        for band in bands:
            if band in image.bandNames().getInfo():
                band_image = image.select(band)
                
                # Find dark object value (1st percentile)
                dark_object = band_image.reduceRegion(
                    reducer=ee.Reducer.percentile([1]),
                    geometry=image.geometry(),
                    scale=sensor_config['scale'],
                    maxPixels=1e9
                ).get(band)
                
                # Subtract dark object value
                corrected_band = band_image.subtract(ee.Number(dark_object))
                corrected_bands.append(corrected_band)
        
        # Combine corrected bands
        corrected_image = ee.Image.cat(corrected_bands)
        
        return corrected_image.copyProperties(image, image.propertyNames())
    
    def topographic_correction(self, image: ee.Image, sensor: str, aoi: ee.Geometry) -> ee.Image:
        """
        Apply topographic correction using Minnaert correction
        
        Args:
            image: Earth Engine image
            sensor: Sensor type
            aoi: Area of interest
            
        Returns:
            Topographically corrected image
        """
        if not self.config['preprocessing']['enable_topographic_correction']:
            return image
        
        # Get DEM (SRTM 30m)
        dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)
        
        # Calculate slope and aspect
        slope = ee.Terrain.slope(dem).multiply(np.pi).divide(180)
        aspect = ee.Terrain.aspect(dem).multiply(np.pi).divide(180)
        
        # Get sun angles from image metadata
        sun_azimuth = ee.Number(image.get('SUN_AZIMUTH')).multiply(np.pi).divide(180)
        sun_elevation = ee.Number(image.get('SUN_ELEVATION')).multiply(np.pi).divide(180)
        
        # Calculate cosine of solar incidence angle
        cos_i = (sun_elevation.cos().multiply(slope.cos())).add(
            sun_elevation.sin().multiply(slope.sin()).multiply(
                sun_azimuth.subtract(aspect).cos()
            )
        )
        
        # Apply Minnaert correction
        sensor_config = self.sensor_configs[sensor]
        corrected_bands = []
        
        for band in sensor_config['bands']:
            if band in image.bandNames().getInfo():
                band_image = image.select(band)
                
                # Minnaert constant (typically 0.5-1.0)
                k = 0.7
                
                # Apply correction
                corrected_band = band_image.multiply(
                    cos_i.divide(slope.cos()).pow(k)
                )
                corrected_bands.append(corrected_band)
        
        corrected_image = ee.Image.cat(corrected_bands)
        return corrected_image.copyProperties(image, image.propertyNames())
    
    def quality_assessment(self, image: ee.Image, sensor: str, aoi: ee.Geometry) -> Dict:
        """
        Comprehensive quality assessment of processed image
        
        Args:
            image: Processed Earth Engine image
            sensor: Sensor type
            aoi: Area of interest
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        # 1. Cloud coverage assessment
        if 'cloud_mask' in image.bandNames().getInfo():
            cloud_mask = image.select('cloud_mask')
            cloud_coverage = cloud_mask.Not().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=self.sensor_configs[sensor]['scale'],
                maxPixels=1e9
            ).get('cloud_mask')
            quality_metrics['cloud_coverage'] = cloud_coverage
        
        # 2. Shadow coverage assessment
        if 'shadow_mask' in image.bandNames().getInfo():
            shadow_mask = image.select('shadow_mask')
            shadow_coverage = shadow_mask.Not().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=self.sensor_configs[sensor]['scale'],
                maxPixels=1e9
            ).get('shadow_mask')
            quality_metrics['shadow_coverage'] = shadow_coverage
        
        # 3. Data availability (non-null pixels)
        valid_pixels = image.select(0).mask().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=self.sensor_configs[sensor]['scale'],
            maxPixels=1e9
        )
        quality_metrics['data_availability'] = list(valid_pixels.values())[0]
        
        # 4. Spectral quality assessment
        if sensor == 'sentinel2':
            red = image.select('B4')
            nir = image.select('B8')
        else:
            red = image.select('SR_B4')
            nir = image.select('SR_B5')
        
        # Calculate NDVI as spectral quality indicator
        ndvi = red.subtract(nir).divide(red.add(nir))
        ndvi_stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
            geometry=aoi,
            scale=self.sensor_configs[sensor]['scale'],
            maxPixels=1e9
        )
        
        # Spectral quality based on NDVI variability
        ndvi_std = ndvi_stats.get('nd_stdDev')
        quality_metrics['spectral_quality'] = ee.Number(1).subtract(
            ee.Number(ndvi_std).divide(0.5)
        ).max(0)
        
        # 5. Overall quality score
        weights = {
            'cloud_coverage': -0.3,
            'shadow_coverage': -0.2,
            'data_availability': 0.3,
            'spectral_quality': 0.2
        }
        
        overall_score = 0.8  # Base score
        for metric, weight in weights.items():
            if metric in quality_metrics:
                overall_score += weight * quality_metrics[metric]
        
        quality_metrics['overall_quality'] = max(0, min(1, overall_score))
        
        return quality_metrics
    
    def process_image_collection(self, collection: ee.ImageCollection, 
                               sensor: str, aoi: ee.Geometry) -> Tuple[ee.Image, Dict]:
        """
        Process entire image collection with quality control
        
        Args:
            collection: Earth Engine image collection
            sensor: Sensor type
            aoi: Area of interest
            
        Returns:
            Tuple of (processed_composite, quality_report)
        """
        self.logger.info(f"Processing {sensor} collection with {collection.size().getInfo()} images")
        
        # Process each image in collection
        processed_collection = collection.map(
            lambda img: self._process_single_image(img, sensor, aoi)
        )
        
        # Quality filtering
        quality_filtered = processed_collection.filter(
            ee.Filter.gt('overall_quality', 0.5)
        )
        
        # Create composite
        composite = quality_filtered.median()
        
        # Final quality assessment
        quality_report = self.quality_assessment(composite, sensor, aoi)
        
        self.logger.info(f"Processed composite with quality score: {quality_report.get('overall_quality', 'N/A')}")
        
        return composite, quality_report
    
    def _process_single_image(self, image: ee.Image, sensor: str, aoi: ee.Geometry) -> ee.Image:
        """Process a single image through the preprocessing pipeline"""
        # Apply cloud masking
        if self.config['preprocessing']['enable_cloud_masking']:
            image = self.advanced_cloud_masking(image, sensor)
        
        # Apply shadow masking
        if self.config['preprocessing']['enable_shadow_masking']:
            image = self.shadow_masking(image, sensor)
        
        # Apply atmospheric correction
        image = self.atmospheric_correction(image, sensor)
        
        # Apply topographic correction
        image = self.topographic_correction(image, sensor, aoi)
        
        # Calculate quality metrics
        quality_metrics = self.quality_assessment(image, sensor, aoi)
        
        # Add quality metrics as image properties
        for metric, value in quality_metrics.items():
            image = image.set(metric, value)
        
        return image
    
    def confidence_scoring(self, before_image: ee.Image, after_image: ee.Image,
                          change_threshold: float = 0.1) -> ee.Image:
        """
        Calculate confidence scores for change detection
        
        Args:
            before_image: Pre-change image
            after_image: Post-change image
            change_threshold: Minimum change threshold
            
        Returns:
            Confidence score image
        """
        # Calculate spectral difference
        diff = after_image.subtract(before_image).abs()
        
        # Calculate temporal stability (inverse of change magnitude)
        stability = diff.reduce(ee.Reducer.mean()).multiply(-1).add(1)
        
        # Calculate spatial consistency
        spatial_variance = diff.reduceNeighborhood(
            reducer=ee.Reducer.variance(),
            kernel=ee.Kernel.square(3)
        )
        spatial_consistency = spatial_variance.multiply(-1).add(1)
        
        # Calculate overall confidence
        confidence = stability.multiply(0.6).add(spatial_consistency.multiply(0.4))
        
        # Apply threshold
        confidence = confidence.where(diff.gt(change_threshold), confidence)
        
        return confidence.rename('confidence')
    
    def false_positive_mitigation(self, change_image: ee.Image, 
                                 confidence_image: ee.Image,
                                 sensor: str) -> ee.Image:
        """
        Mitigate false positives using multiple approaches
        
        Args:
            change_image: Change detection result
            confidence_image: Confidence scores
            sensor: Sensor type
            
        Returns:
            Filtered change image
        """
        # 1. Confidence threshold filtering
        high_confidence = confidence_image.gt(0.7)
        
        # 2. Morphological operations to remove noise
        cleaned_change = change_image.updateMask(high_confidence)
        
        # Opening operation (erosion followed by dilation)
        kernel = ee.Kernel.circle(radius=1)
        eroded = cleaned_change.focal_min(kernel=kernel)
        opened = eroded.focal_max(kernel=kernel)
        
        # 3. Minimum mapping unit filtering
        # Remove small patches (< 0.1 hectares)
        min_area = 1000  # square meters
        scale = self.sensor_configs[sensor]['scale']
        
        # Convert to binary
        binary_change = opened.gt(0)
        
        # Label connected components
        labeled = binary_change.connectedComponents(
            connectedness=ee.Kernel.plus(1),
            maxSize=256
        )
        
        # Calculate area of each component
        area_image = labeled.select('labels').connectedPixelCount(
            maxSize=1000, eightConnected=True
        ).multiply(scale * scale)
        
        # Filter by minimum area
        large_changes = area_image.gte(min_area)
        filtered_change = opened.updateMask(large_changes)
        
        return filtered_change
    
    def export_quality_report(self, quality_metrics: Dict, 
                            output_path: str) -> None:
        """
        Export quality assessment report
        
        Args:
            quality_metrics: Quality metrics dictionary
            output_path: Output file path
        """
        report = {
            'preprocessing_report': {
                'timestamp': datetime.now().isoformat(),
                'quality_metrics': quality_metrics,
                'thresholds': self.quality_thresholds,
                'configuration': self.config
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Quality report exported to {output_path}")


class LocalDataPreprocessor:
    """
    Preprocessor for locally exported satellite data.
    Enhanced with advanced preprocessing capabilities.
    """
    
    def __init__(self, data_dir: str = "data/exports", config_path: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing exported satellite data
            config_path: Path to preprocessing configuration file
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)    
    def load_satellite_image(self, file_path: str) -> Dict:
        """
        Load satellite image and extract metadata.
        
        Args:
            file_path: Path to GeoTIFF file
            
        Returns:
            Dict: Image data and metadata
        """
        try:
            with rasterio.open(file_path) as src:
                # Read all bands
                data = src.read()
                
                # Get metadata
                metadata = {
                    'file_path': file_path,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'resolution': src.res,
                    'band_names': self._get_band_names(src.count)
                }
                
                print(f"✅ Loaded: {os.path.basename(file_path)}")
                print(f"   📐 Size: {metadata['width']} x {metadata['height']}")
                print(f"   🔢 Bands: {metadata['count']}")
                print(f"   📏 Resolution: {metadata['resolution'][0]:.1f}m")
                
                return {
                    'data': data,
                    'metadata': metadata
                }
                
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            return None
        """
        Load satellite image and extract metadata.
        
        Args:
            file_path: Path to GeoTIFF file
            
        Returns:
            Dict: Image data and metadata
        """
        try:
            with rasterio.open(file_path) as src:
                # Read all bands
                data = src.read()
                
                # Get metadata
                metadata = {
                    'file_path': file_path,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'resolution': src.res,
                    'band_names': self.advanced_preprocessor._get_band_names(src.count)
                }
                
                print(f"✅ Loaded: {os.path.basename(file_path)}")
                print(f"   📐 Size: {metadata['width']} x {metadata['height']}")
                print(f"   🔢 Bands: {metadata['count']}")
                print(f"   📏 Resolution: {metadata['resolution'][0]:.1f}m")
                
                return {
                    'data': data,
                    'metadata': metadata
                }
                
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            return None
    
    def preprocess_image(self, image_path: str) -> Dict:
        """
        Preprocess a single image for analysis.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict: Processed image data and metadata
        """
        print(f"🔄 Preprocessing image: {os.path.basename(image_path)}")
        
        # Load image
        img_data = self.load_satellite_image(image_path)
        
        if not img_data:
            print("❌ Image loading failed")
            return None
        
        # Normalize reflectance values
        normalized = self.advanced_preprocessor.normalize_reflectance_values(
            img_data['data'], 'sentinel2'
        )
        
        # Calculate additional indices
        indices = self.advanced_preprocessor.calculate_additional_indices(
            normalized, img_data['metadata']['band_names']
        )
        
        # Update metadata with new indices
        img_data['metadata'].update({
            'indices': indices
        })
        
        # Save processed image
        output_path = self.data_dir / f"processed_{Path(image_path).name}"
        self._save_processed_image(normalized, img_data['metadata'], output_path)
        
        print(f"✅ Image processed and saved: {output_path.name}")
        return {
            'data': normalized,
            'metadata': img_data['metadata'],
            'file_path': str(output_path)
        }
    
    def _save_processed_image(self, data: np.ndarray, metadata: Dict, file_path: Path):
        """
        Save processed image data to file.
        
        Args:
            data: Image data array
            metadata: Image metadata
            file_path: Output file path
        """
        # Create directory if not exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as GeoTIFF
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=metadata['height'],
            width=metadata['width'],
            count=len(metadata['band_names']),
            dtype='float32',
            crs=metadata['crs'],
            transform=metadata['transform'],
            compress='lzw'
        ) as dst:
            # Write each band
            for i, band_name in enumerate(metadata['band_names'], start=1):
                dst.write(data[i-1], i)
        
        print(f"✅ Saved processed image: {file_path.name}")
    
    def batch_preprocess(self, pattern: str = "*.tif"):
        """
        Batch preprocess all images matching the pattern in the data directory.
        
        Args:
            pattern: Filename pattern to match (default: "*.tif")
        """
        print(f"📂 Batch preprocessing images: {pattern}")
        
        # Find all image files matching the pattern
        image_files = list(self.data_dir.glob(pattern))
        
        if not image_files:
            print("❌ No images found for preprocessing")
            return
        
        # Process each image
        for image_path in image_files:
            self.preprocess_image(str(image_path))
        
        print("✅ Batch preprocessing complete")
    

def demo_preprocessing():
    """
    Demonstrate preprocessing workflow.
    """
    print("🔧 PREPROCESSING DEMO")
    print("=" * 40)
    
    preprocessor = LocalDataPreprocessor()
    
    # Look for exported files
    export_dir = Path("data/exports")
    tiff_files = list(export_dir.glob("*.tif"))
    
    if len(tiff_files) < 1:
        print("❌ No .tif files found in data/exports/")
        print("💡 Run local_export_demo.py first to export some data")
        return
    
    print(f"📁 Found {len(tiff_files)} files:")
    for i, file_path in enumerate(tiff_files):
        print(f"   {i+1}. {file_path.name}")
    
    # Process first file
    if len(tiff_files) >= 1:
        print(f"\n🔄 Processing: {tiff_files[0].name}")
        img_data = preprocessor.load_satellite_image(str(tiff_files[0]))
        
        if img_data:
            # Normalize
            normalized = preprocessor.normalize_reflectance_values(img_data['data'], 'sentinel2')
            
            # Calculate indices
            indices = preprocessor.calculate_additional_indices(
                normalized, img_data['metadata']['band_names']
            )
            
            # Create visualizations
            rgb = preprocessor.create_rgb_composite(
                normalized, img_data['metadata']['band_names']
            )
            
            if rgb is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(rgb)
                plt.title(f"RGB Composite: {tiff_files[0].name}")
                plt.axis('off')
                plt.tight_layout()
                
                save_path = f"data/exports/{tiff_files[0].stem}_rgb_preview.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📊 RGB preview saved: {save_path}")
                plt.show()
    
    # If we have 2+ files, demo change detection prep
    if len(tiff_files) >= 2:
        print(f"\n🔄 Preparing change detection with 2 files...")
        result = preprocessor.prepare_for_change_detection(
            str(tiff_files[0]), str(tiff_files[1])
        )
        
        if result:
            print("✅ Change detection preparation complete")
            preprocessor.visualize_comparison(
                str(tiff_files[0]), str(tiff_files[1]),
                "data/exports/comparison_preview.png"
            )
    
    # Demo batch preprocessing
    print("\n📂 Demo batch preprocessing")
    batch_dir = Path("data/batch_exports")
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy some TIFF files for batch demo
    for file_path in tiff_files[:2]:
        new_path = batch_dir / file_path.name
        if not new_path.exists():
            os.symlink(file_path, new_path)
            print(f"🔗 Linked: {new_path.name}")
    
    # Run batch preprocessing
    batch_preprocessor = LocalDataPreprocessor(str(batch_dir))
    batch_preprocessor.batch_preprocess("*.tif")
