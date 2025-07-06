"""
Google Earth Engine Data Retrieval for Change Detection System
============================================================

This script demonstrates how to set up and fetch satellite data from Google Earth Engine
for the change detection system.

Requirements:
1. Google Earth Engine account
2. Python environment with required packages
3. Service account credentials (recommended) or OAuth

Setup Instructions:
1. Install required packages:
   pip install earthengine-api
   pip install geemap
   pip install geopandas
   pip install rasterio
   pip install matplotlib
   pip install folium

2. Create GEE account at: https://earthengine.google.com/
3. Set up authentication (see authentication section below)
"""

import ee
import geemap
import geopandas as gpd
import json
import os
from datetime import datetime, timedelta
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import requests
import zipfile
import yaml
from pathlib import Path

class GEEDataRetriever:
    """
    A class to handle Google Earth Engine data retrieval for change detection.
    """
    
    def __init__(self, service_account_path: Optional[str] = None, config_path: Optional[str] = None, preprocessing_config_path: Optional[str] = None):
        """
        Initialize the GEE Data Retriever.
        
        Args:
            service_account_path: Path to service account JSON file (optional)
            config_path: Path to configuration YAML file (optional)
            preprocessing_config_path: Path to preprocessing configuration YAML file (optional)
        """
        self.service_account_path = service_account_path
        self.config = self.load_config(config_path)
        self.preprocessing_config = self.load_preprocessing_config(preprocessing_config_path)
        self.ensure_directories()
        self.initialize_gee()
    
    def initialize_gee(self):
        """
        Initialize Google Earth Engine authentication and API.
        """
        try:
            # Get project ID from config
            project_id = self.config.get('gee', {}).get('project_id')
            
            if (self.service_account_path and 
                isinstance(self.service_account_path, str) and 
                os.path.exists(self.service_account_path)):
                # Service account authentication (recommended for production)
                credentials = ee.ServiceAccountCredentials(
                    None, self.service_account_path
                )
                ee.Initialize(credentials, project=project_id)
                print(f"✅ GEE initialized with service account (Project: {project_id})")
            else:
                # OAuth authentication (for development)
                try:
                    ee.Initialize(project=project_id)
                    print(f"✅ GEE initialized with existing credentials (Project: {project_id})")
                except:
                    print("🔐 First time setup - please authenticate...")
                    ee.Authenticate()  # This will open browser for first-time setup
                    ee.Initialize(project=project_id)
                    print(f"✅ GEE initialized with OAuth (Project: {project_id})")
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            print("Please run: earthengine authenticate")
            raise
    
    def define_aoi(self, coordinates: List[List[float]], name: str = "AOI") -> ee.Geometry:
        """
        Define Area of Interest (AOI) from coordinates.
        
        Args:
            coordinates: List of [lon, lat] coordinates defining the polygon
            name: Name for the AOI
            
        Returns:
            ee.Geometry: Earth Engine geometry object
        """
        # Create polygon geometry
        aoi = ee.Geometry.Polygon(coordinates)
        
        # Validate geometry
        area = aoi.area().getInfo()
        print(f"📍 AOI '{name}' created with area: {area/1000000:.2f} km²")
        
        return aoi
    
    def get_satellite_collection(self, 
                               satellite: str,
                               start_date: str,
                               end_date: str,
                               aoi: ee.Geometry,
                               cloud_coverage: int = None) -> ee.ImageCollection:
        """
        Get satellite image collection for specified parameters.
        
        Args:
            satellite: 'sentinel2' or 'landsat8' or 'landsat9'
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            aoi: Area of Interest geometry
            cloud_coverage: Maximum cloud coverage percentage
            
        Returns:
            ee.ImageCollection: Filtered image collection
        """
        # Get satellite config
        satellite_config = self.config.get('satellites', {}).get(satellite, {})
        if not satellite_config:
            raise ValueError(f"Satellite '{satellite}' not configured")
        
        collection_id = satellite_config['collection_id']
        if cloud_coverage is None:
            cloud_coverage = satellite_config['cloud_coverage_threshold']
        
        # Get collection
        collection = ee.ImageCollection(collection_id)
        
        # Filter collection
        filtered_collection = (collection
                             .filterBounds(aoi)
                             .filterDate(start_date, end_date)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_coverage))
                             .sort('system:time_start'))
        
        count = filtered_collection.size().getInfo()
        print(f"📡 Found {count} {satellite} images for date range {start_date} to {end_date}")
        
        
        return filtered_collection
    
    def create_advanced_cloud_mask(self, image: ee.Image, satellite: str) -> ee.Image:
        """
        Create advanced cloud mask using preprocessing configuration.
        
        Args:
            image: Input image
            satellite: Satellite type
            
        Returns:
            ee.Image: Cloud-masked image with quality assessment
        """
        preprocessing = self.preprocessing_config.get('preprocessing', {})
        sensor_config = self.preprocessing_config.get('sensors', {}).get(satellite, {})
        
        if not preprocessing.get('enable_cloud_masking', True):
            return image
        
        if satellite == 'sentinel2':
            # Enhanced Sentinel-2 cloud masking
            cloud_method = sensor_config.get('cloud_detection_method', 'qa_bands')
            cloud_threshold = sensor_config.get('cloud_probability_threshold', 60)
            
            if cloud_method == 'multi_method':
                # Use multiple methods for better accuracy
                qa = image.select('QA60')
                cloudBitMask = 1 << 10
                cirrusBitMask = 1 << 11
                qa_mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                         qa.bitwiseAnd(cirrusBitMask).eq(0))
                
                # Additional spectral cloud detection
                blue = image.select('B2')
                green = image.select('B3')
                red = image.select('B4')
                nir = image.select('B8')
                
                # Simple cloud probability based on spectral characteristics
                spectral_mask = blue.gt(0.3).And(green.gt(0.3)).And(red.gt(0.3)).And(nir.gt(0.3))
                
                mask = qa_mask.And(spectral_mask.Not())
            else:
                # Standard QA band masking
                qa = image.select('QA60')
                cloudBitMask = 1 << 10
                cirrusBitMask = 1 << 11
                mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
                       qa.bitwiseAnd(cirrusBitMask).eq(0))
            
            # Apply cloud buffer if configured
            cloud_buffer = preprocessing.get('cloud_buffer_distance', 0)
            if cloud_buffer > 0:
                mask = mask.focal_min(cloud_buffer, 'circle', 'meters')
            
            masked_image = image.updateMask(mask)
            
        elif satellite in ['landsat8', 'landsat9']:
            # Enhanced Landsat cloud masking
            qa = image.select('QA_PIXEL')
            cloudShadowBitMask = 1 << 3
            cloudsBitMask = 1 << 5
            
            mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
                   qa.bitwiseAnd(cloudsBitMask).eq(0))
            
            # Apply cloud buffer
            cloud_buffer = preprocessing.get('cloud_buffer_distance', 0)
            if cloud_buffer > 0:
                mask = mask.focal_min(cloud_buffer, 'circle', 'meters')
            
            masked_image = image.updateMask(mask)
        else:
            masked_image = image
        
        # Apply shadow masking if enabled
        if preprocessing.get('enable_shadow_masking', True):
            masked_image = self.create_shadow_mask(masked_image, satellite)
        
        return masked_image
    
    def create_shadow_mask(self, image: ee.Image, satellite: str) -> ee.Image:
        """
        Create shadow mask based on preprocessing configuration.
        
        Args:
            image: Input image
            satellite: Satellite type
            
        Returns:
            ee.Image: Shadow-masked image
        """
        sensor_config = self.preprocessing_config.get('sensors', {}).get(satellite, {})
        shadow_method = sensor_config.get('shadow_detection_method', 'spectral')
        
        if satellite == 'sentinel2' and shadow_method == 'spectral_topographic':
            # Enhanced shadow detection for Sentinel-2
            nir = image.select('B8')
            swir1 = image.select('B11')
            
            # Simple shadow detection based on low NIR and SWIR values
            shadow_mask = nir.gt(0.15).And(swir1.gt(0.1))
            
            # Apply shadow buffer
            shadow_buffer = self.preprocessing_config.get('preprocessing', {}).get('shadow_buffer_distance', 0)
            if shadow_buffer > 0:
                shadow_mask = shadow_mask.focal_min(shadow_buffer, 'circle', 'meters')
            
            return image.updateMask(shadow_mask)
        
        return image
    
    def create_cloud_mask(self, image: ee.Image, satellite: str) -> ee.Image:
        """
        Create cloud mask for different satellite types (legacy method).
        """
        return self.create_advanced_cloud_mask(image, satellite)
    
    def get_band_mapping(self, satellite: str) -> Dict[str, str]:
        """
        Get band mapping for different satellites to standardize names.
        
        Args:
            satellite: Satellite type
            
        Returns:
            Dict: Band mapping
        """
        satellite_config = self.config.get('satellites', {}).get(satellite, {})
        return satellite_config.get('bands', {})
    
    def calculate_spectral_indices(self, image: ee.Image, satellite: str) -> ee.Image:
        """
        Calculate spectral indices based on preprocessing configuration.
        
        Args:
            image: Input image with standardized band names
            satellite: Satellite type
            
        Returns:
            ee.Image: Image with calculated indices
        """
        preprocessing = self.preprocessing_config.get('preprocessing', {})
        if not preprocessing.get('enable_spectral_indices', True):
            return image
        
        indices_to_calc = preprocessing.get('indices_to_calculate', ['NDVI', 'NDWI', 'NDBI'])
        sensor_config = self.preprocessing_config.get('sensors', {}).get(satellite, {})
        indices_config = sensor_config.get('indices', {})
        
        calculated_indices = []
        
        for index_name in indices_to_calc:
            if index_name in indices_config:
                index_config = indices_config[index_name]
                required_bands = index_config.get('bands', [])
                
                # Check if all required bands are available
                available_bands = image.bandNames().getInfo()
                if all(band in available_bands for band in required_bands):
                    try:
                        if index_name == 'NDVI':
                            if satellite == 'sentinel2':
                                index = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
                            else:  # Landsat
                                index = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi')
                        elif index_name == 'NDWI':
                            if satellite == 'sentinel2':
                                index = image.normalizedDifference(['B3', 'B8']).rename('ndwi')
                            else:  # Landsat
                                index = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi')
                        elif index_name == 'NDBI':
                            if satellite == 'sentinel2':
                                index = image.normalizedDifference(['B11', 'B8']).rename('ndbi')
                            else:  # Landsat
                                index = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('ndbi')
                        elif index_name == 'EVI':
                            formula = index_config.get('formula', '')
                            if satellite == 'sentinel2':
                                index = image.expression(
                                    '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
                                    {'B8': image.select('B8'), 'B4': image.select('B4'), 'B2': image.select('B2')}
                                ).rename('evi')
                            else:  # Landsat
                                index = image.expression(
                                    '2.5 * ((B5 - B4) / (B5 + 6 * B4 - 7.5 * B2 + 1))',
                                    {'B5': image.select('SR_B5'), 'B4': image.select('SR_B4'), 'B2': image.select('SR_B2')}
                                ).rename('evi')
                        elif index_name == 'SAVI':
                            if satellite == 'sentinel2':
                                index = image.expression(
                                    '((B8 - B4) / (B8 + B4 + 0.5)) * 1.5',
                                    {'B8': image.select('B8'), 'B4': image.select('B4')}
                                ).rename('savi')
                            else:  # Landsat
                                index = image.expression(
                                    '((B5 - B4) / (B5 + B4 + 0.5)) * 1.5',
                                    {'B5': image.select('SR_B5'), 'B4': image.select('SR_B4')}
                                ).rename('savi')
                        elif index_name == 'MSAVI':
                            if satellite == 'sentinel2':
                                index = image.expression(
                                    '(2 * B8 + 1 - sqrt(pow(2 * B8 + 1, 2) - 8 * (B8 - B4))) / 2',
                                    {'B8': image.select('B8'), 'B4': image.select('B4')}
                                ).rename('msavi')
                            else:  # Landsat
                                index = image.expression(
                                    '(2 * B5 + 1 - sqrt(pow(2 * B5 + 1, 2) - 8 * (B5 - B4))) / 2',
                                    {'B5': image.select('SR_B5'), 'B4': image.select('SR_B4')}
                                ).rename('msavi')
                        else:
                            continue
                        
                        calculated_indices.append(index)
                        
                    except Exception as e:
                        print(f"⚠️ Failed to calculate {index_name}: {e}")
                        continue
        
        # Add calculated indices to image
        if calculated_indices:
            return image.addBands(calculated_indices)
        else:
            return image
    
    def calculate_indices(self, image: ee.Image) -> ee.Image:
        """
        Legacy method for calculating indices - redirects to new method.
        """
        # Try to detect satellite type from band names
        bands = image.bandNames().getInfo()
        if 'B4' in bands:
            satellite = 'sentinel2'
        elif 'SR_B4' in bands:
            satellite = 'landsat8'  # Could be landsat9 too, but same bands
        else:
            satellite = 'sentinel2'  # Default
        
        return self.calculate_spectral_indices(image, satellite)
    
    def apply_quality_control(self, image: ee.Image, satellite: str) -> Tuple[ee.Image, Dict]:
        """
        Apply quality control based on preprocessing configuration.
        """
        quality_config = self.preprocessing_config.get('quality_control', {})
        
        if not quality_config.get('enable_outlier_detection', True):
            return image, {}
        
        quality_metrics = {}
        
        # Basic quality assessment using available bands
        try:
            # Get available bands
            available_bands = image.bandNames().getInfo()
            
            # Calculate data availability using first available band
            if available_bands:
                valid_pixels = image.select(available_bands[0]).mask().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    scale=100,
                    maxPixels=1e6
                ).getInfo()
                quality_metrics['data_availability'] = list(valid_pixels.values())[0] if valid_pixels else 0
            
            # For standardized images, skip cloud assessment
            quality_metrics['cloud_coverage'] = 0  # Already processed
            
        except Exception as e:
            print(f"⚠️ Quality assessment failed: {e}")
            quality_metrics = {'quality_check': 'failed'}
        
        return image, quality_metrics

    def process_image_collection(self, 
                               collection: ee.ImageCollection,
                               satellite: str,
                               aoi: ee.Geometry) -> ee.Image:
        """
        Process image collection to create median composite with advanced preprocessing.
        
        Args:
            collection: Image collection
            satellite: Satellite type
            aoi: Area of Interest
            
        Returns:
            ee.Image: Processed composite image
        """
        # Apply advanced cloud masking
        def mask_clouds_and_shadows(image):
            return self.create_advanced_cloud_mask(image, satellite)
        
        masked_collection = collection.map(mask_clouds_and_shadows)
        
        # Temporal filtering based on preprocessing config
        temporal_config = self.preprocessing_config.get('preprocessing', {})
        min_images = temporal_config.get('min_images_per_composite', 2)
        max_images = temporal_config.get('max_images_per_composite', 10)
        
        # Limit collection size if needed
        collection_size = masked_collection.size().getInfo()
        if collection_size > max_images:
            masked_collection = masked_collection.limit(max_images)
        elif collection_size < min_images:
            print(f"⚠️ Warning: Only {collection_size} images available (minimum: {min_images})")
        
        # Create median composite
        composite = masked_collection.median().clip(aoi)
        
        # Select and rename bands
        band_mapping = self.get_band_mapping(satellite)
        if band_mapping:
            bands = list(band_mapping.values())
            new_names = list(band_mapping.keys())
            try:
                composite = composite.select(bands, new_names)
            except:
                # If some bands are missing, select available ones
                available_bands = composite.bandNames().getInfo()
                valid_bands = [b for b in bands if b in available_bands]
                valid_names = [new_names[bands.index(b)] for b in valid_bands]
                if valid_bands:
                    composite = composite.select(valid_bands, valid_names)
        
        # Apply quality control
        composite, quality_metrics = self.apply_quality_control(composite, satellite)
        
        # Calculate spectral indices
        composite = self.calculate_spectral_indices(composite, satellite)
        
        return composite
    
    def load_preprocessing_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load preprocessing configuration from YAML file.
        
        Args:
            config_path: Path to preprocessing config file
            
        Returns:
            Dict: Preprocessing configuration dictionary
        """
        default_preprocessing_config = {
            'preprocessing': {
                'enable_cloud_masking': True,
                'enable_shadow_masking': True,
                'enable_spectral_indices': True,
                'indices_to_calculate': ['NDVI', 'NDWI', 'NDBI', 'EVI']
            },
            'quality_control': {
                'enable_outlier_detection': True,
                'outlier_threshold': 2.5
            },
            'sensors': {
                'sentinel2': {
                    'cloud_detection_method': 'qa_bands',
                    'cloud_probability_threshold': 60
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config
            except Exception as e:
                print(f"⚠️ Failed to load preprocessing config: {e}, using defaults")
        elif not config_path:
            # Try to find preprocessing config automatically
            possible_paths = [
                'config/preprocessing_config.yaml',
                '../config/preprocessing_config.yaml'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            config = yaml.safe_load(f)
                        print(f"📄 Loaded preprocessing config from: {path}")
                        return config
                    except Exception as e:
                        continue
        
        print("📄 Using default preprocessing configuration")
        return default_preprocessing_config
    
    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dict: Configuration dictionary
        """
        default_config = {
            'gee': {
                'max_pixels': 1000000000,
                'default_scale': 10
            },
            'satellites': {
                'sentinel2': {
                    'collection_id': 'COPERNICUS/S2_SR_HARMONIZED',
                    'cloud_coverage_threshold': 20,
                    'bands': {
                        'red': 'B4',
                        'green': 'B3',
                        'blue': 'B2',
                        'nir': 'B8',
                        'swir1': 'B11',
                        'swir2': 'B12'
                    }
                },
                'landsat8': {
                    'collection_id': 'LANDSAT/LC08/C02/T1_L2',
                    'cloud_coverage_threshold': 20,
                    'bands': {
                        'red': 'SR_B4',
                        'green': 'SR_B3',
                        'blue': 'SR_B2',
                        'nir': 'SR_B5',
                        'swir1': 'SR_B6',
                        'swir2': 'SR_B7'
                    }
                },
                'landsat9': {
                    'collection_id': 'LANDSAT/LC09/C02/T1_L2',
                    'cloud_coverage_threshold': 20,
                    'bands': {
                        'red': 'SR_B4',
                        'green': 'SR_B3',
                        'blue': 'SR_B2',
                        'nir': 'SR_B5',
                        'swir1': 'SR_B6',
                        'swir2': 'SR_B7'
                    }
                }
            },
            'export': {
                'local_download_path': 'data/exports/',
                'default_format': 'GeoTIFF'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                # Merge with default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                            elif isinstance(subvalue, dict):
                                for subsubkey, subsubvalue in subvalue.items():
                                    if subsubkey not in config[key][subkey]:
                                        config[key][subkey][subsubkey] = subsubvalue
                return config
            except Exception as e:
                print(f"⚠️  Failed to load config: {e}, using defaults")
        
        return default_config
    
    def ensure_directories(self):
        """
        Ensure required directories exist.
        """
        dirs_to_create = [
            'data/raw',
            'data/processed', 
            'data/exports',
            'logs',
            self.config.get('export', {}).get('local_download_path', 'data/exports/')
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("📁 Directory structure verified")

def main():
    """
    Main function demonstrating the integrated data retrieval process.
    """
    print("🌍 Google Earth Engine Change Detection - Integrated Data Retrieval Demo")
    print("=" * 70)
    
    # Initialize data retriever with both configs
    retriever = GEEDataRetriever(
        config_path="config/config.yaml",
        preprocessing_config_path="config/preprocessing_config.yaml"
    )
    
    aoi_coords = [[
        [88.35, 22.55],
        [88.36, 22.55],
        [88.36, 22.56],
        [88.35, 22.56],
        [88.35, 22.55]
    ]]
    aoi = retriever.define_aoi(aoi_coords, "Kolkata")
    
    # Get satellite collection
    collection = retriever.get_satellite_collection(
        'sentinel2', "2023-01-01", "2023-01-31", aoi
    )
    
    # Process collection with advanced preprocessing
    print("\n🔄 Processing with advanced preprocessing...")
    composite = retriever.process_image_collection(collection, 'sentinel2', aoi)
    
    print("✅ Integrated data retrieval with preprocessing complete!")
    print("📊 Features enabled:")
    print("   - Advanced cloud/shadow masking")
    print("   - Quality control assessment")
    print("   - Comprehensive spectral indices")
    print("   - Configurable preprocessing pipeline")

if __name__ == "__main__":
    main()