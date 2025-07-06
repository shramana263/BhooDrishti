import ee
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

class ChangeDetectionEngine:
    """
    Advanced change detection engine using Google Earth Engine
    Implements NDVI analysis, spectral indices, and classification methods
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Change detection thresholds
        self.ndvi_threshold = config.get('change_detection', {}).get('ndvi_threshold', 0.1)
        self.confidence_threshold = config.get('change_detection', {}).get('confidence_threshold', 0.5)
        
    def calculate_spectral_indices(self, image: ee.Image) -> ee.Image:
        """
        Calculate various spectral indices for change detection
        """
        try:
            # Get available bands
            available_bands = image.bandNames().getInfo()
            indices = []
            
            # NDVI (Normalized Difference Vegetation Index)
            if 'nir' in available_bands and 'red' in available_bands:
                ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI')
                indices.append(ndvi)
            
            # NDBI (Normalized Difference Built-up Index)
            if 'swir1' in available_bands and 'nir' in available_bands:
                ndbi = image.normalizedDifference(['swir1', 'nir']).rename('NDBI')
                indices.append(ndbi)
            
            # MNDWI (Modified Normalized Difference Water Index)
            if 'green' in available_bands and 'swir1' in available_bands:
                mndwi = image.normalizedDifference(['green', 'swir1']).rename('MNDWI')
                indices.append(mndwi)
            
            # EVI (Enhanced Vegetation Index)
            if all(band in available_bands for band in ['nir', 'red', 'blue']):
                evi = image.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                    {
                        'NIR': image.select('nir'),
                        'RED': image.select('red'),
                        'BLUE': image.select('blue')
                    }
                ).rename('EVI')
                indices.append(evi)
            
            # SAVI (Soil Adjusted Vegetation Index)
            if 'nir' in available_bands and 'red' in available_bands:
                savi = image.expression(
                    '((NIR - RED) / (NIR + RED + 0.5)) * (1 + 0.5)',
                    {
                        'NIR': image.select('nir'),
                        'RED': image.select('red')
                    }
                ).rename('SAVI')
                indices.append(savi)
            
            # NBR (Normalized Burn Ratio)
            if 'nir' in available_bands and 'swir2' in available_bands:
                nbr = image.normalizedDifference(['nir', 'swir2']).rename('NBR')
                indices.append(nbr)
            
            # Add all calculated indices to the image
            if indices:
                return image.addBands(indices)
            else:
                return image
            
        except Exception as e:
            self.logger.error(f"Error calculating spectral indices: {str(e)}")
            raise
    
    def detect_vegetation_changes(self, before_image: ee.Image, after_image: ee.Image, 
                                aoi: ee.Geometry) -> Dict:
        """
        Detect vegetation changes using NDVI analysis
        """
        try:
            # Check if NIR and red bands are available
            before_bands = before_image.bandNames().getInfo()
            after_bands = after_image.bandNames().getInfo()
            
            if not ('nir' in before_bands and 'red' in before_bands):
                raise ValueError(f"NIR or red band missing. Available bands: {before_bands}")
            
            # Calculate NDVI for both images
            before_ndvi = before_image.normalizedDifference(['nir', 'red']).rename('NDVI_before')
            after_ndvi = after_image.normalizedDifference(['nir', 'red']).rename('NDVI_after')
            
            # Calculate NDVI difference
            ndvi_diff = after_ndvi.subtract(before_ndvi).rename('NDVI_diff')
            
            # Classify changes
            deforestation = ndvi_diff.lt(-self.ndvi_threshold).rename('deforestation')
            afforestation = ndvi_diff.gt(self.ndvi_threshold).rename('afforestation')
            
            # Calculate change statistics
            stats = {
                'deforestation_area': deforestation.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                ),
                'afforestation_area': afforestation.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                ),
                'mean_ndvi_change': ndvi_diff.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                )
            }
            
            return {
                'change_image': ndvi_diff,
                'deforestation_mask': deforestation,
                'afforestation_mask': afforestation,
                'statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting vegetation changes: {str(e)}")
            raise
    
    def detect_urban_expansion(self, before_image: ee.Image, after_image: ee.Image, 
                             aoi: ee.Geometry) -> Dict:
        """
        Detect urban expansion using NDBI and spectral analysis
        """
        try:
            # Check if required bands are available
            before_bands = before_image.bandNames().getInfo()
            after_bands = after_image.bandNames().getInfo()
            
            # Calculate NDBI if SWIR and NIR are available
            if 'swir1' in before_bands and 'nir' in before_bands:
                before_ndbi = before_image.normalizedDifference(['swir1', 'nir']).rename('NDBI_before')
                after_ndbi = after_image.normalizedDifference(['swir1', 'nir']).rename('NDBI_after')
                
                # Calculate NDBI difference
                ndbi_diff = after_ndbi.subtract(before_ndbi).rename('NDBI_diff')
                
                # Urban expansion threshold
                urban_threshold = 0.05
                urban_expansion = ndbi_diff.gt(urban_threshold).rename('urban_expansion')
                
                # Additional criteria: decrease in NDVI if NIR and red are available
                if 'nir' in before_bands and 'red' in before_bands:
                    before_ndvi = before_image.normalizedDifference(['nir', 'red'])
                    after_ndvi = after_image.normalizedDifference(['nir', 'red'])
                    ndvi_decrease = before_ndvi.subtract(after_ndvi).gt(0.1)
                    
                    # Combined urban expansion mask
                    urban_change = urban_expansion.And(ndvi_decrease).rename('urban_change')
                else:
                    # Use only NDBI if NDVI cannot be calculated
                    urban_change = urban_expansion.rename('urban_change')
            else:
                # Fallback: use simple brightness increase if SWIR not available
                if all(band in before_bands for band in ['red', 'green', 'blue']):
                    before_brightness = before_image.select(['red', 'green', 'blue']).reduce(ee.Reducer.mean())
                    after_brightness = after_image.select(['red', 'green', 'blue']).reduce(ee.Reducer.mean())
                    brightness_increase = after_brightness.subtract(before_brightness).gt(0.05)
                    urban_change = brightness_increase.rename('urban_change')
                    ndbi_diff = ee.Image.constant(0).rename('NDBI_diff')
                else:
                    raise ValueError(f"Insufficient bands for urban detection. Available: {before_bands}")
            
            # Calculate statistics
            stats = {
                'urban_expansion_area': urban_change.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                ),
                'mean_ndbi_change': ndbi_diff.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                )
            }
            
            return {
                'change_image': ndbi_diff,
                'urban_expansion_mask': urban_change,
                'statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting urban expansion: {str(e)}")
            raise
    
    def detect_water_body_changes(self, before_image: ee.Image, after_image: ee.Image, 
                                 aoi: ee.Geometry) -> Dict:
        """
        Detect water body changes using MNDWI
        """
        try:
            # Check if required bands are available
            before_bands = before_image.bandNames().getInfo()
            after_bands = after_image.bandNames().getInfo()
            
            # Calculate MNDWI if green and SWIR1 are available
            if 'green' in before_bands and 'swir1' in before_bands:
                before_mndwi = before_image.normalizedDifference(['green', 'swir1']).rename('MNDWI_before')
                after_mndwi = after_image.normalizedDifference(['green', 'swir1']).rename('MNDWI_after')
            else:
                # Fallback: use NDWI (green - NIR) if SWIR1 not available
                if 'green' in before_bands and 'nir' in before_bands:
                    before_mndwi = before_image.normalizedDifference(['green', 'nir']).rename('NDWI_before')
                    after_mndwi = after_image.normalizedDifference(['green', 'nir']).rename('NDWI_after')
                else:
                    raise ValueError(f"Insufficient bands for water detection. Available: {before_bands}")
            
            # Water masks (MNDWI > 0 indicates water)
            before_water = before_mndwi.gt(0).rename('water_before')
            after_water = after_mndwi.gt(0).rename('water_after')
            
            # Calculate changes
            water_loss = before_water.And(after_water.Not()).rename('water_loss')
            water_gain = before_water.Not().And(after_water).rename('water_gain')
            
            # Calculate statistics
            stats = {
                'water_loss_area': water_loss.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                ),
                'water_gain_area': water_gain.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=aoi,
                    scale=10,
                    maxPixels=1e9
                )
            }
            
            return {
                'water_loss_mask': water_loss,
                'water_gain_mask': water_gain,
                'before_water_mask': before_water,
                'after_water_mask': after_water,
                'statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting water body changes: {str(e)}")
            raise
    
    def classify_land_cover_change(self, before_image: ee.Image, after_image: ee.Image, 
                                  aoi: ee.Geometry) -> Dict:
        """
        Classify land cover changes using spectral analysis
        """
        try:
            # Add spectral indices to both images
            before_processed = self.calculate_spectral_indices(before_image)
            after_processed = self.calculate_spectral_indices(after_image)
            
            # Get available bands
            available_bands = before_processed.bandNames().getInfo()
            
            # Create change image with available bands
            change_bands = []
            for band in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']:
                if band in available_bands:
                    diff_band = after_processed.select(band).subtract(
                        before_processed.select(band)
                    ).rename(f'{band}_diff')
                    change_bands.append(diff_band)
            
            # Add index differences if available
            for index in ['NDVI', 'NDBI', 'MNDWI']:
                if index in available_bands:
                    diff_band = after_processed.select(index).subtract(
                        before_processed.select(index)
                    ).rename(f'{index}_diff')
                    change_bands.append(diff_band)
            
            if not change_bands:
                raise ValueError("No bands available for change classification")
            
            # Combine all change bands
            change_image = ee.Image.cat(change_bands)
            
            # Simple threshold-based classification
            classification = ee.Image(0)  # No change
            
            # Vegetation loss (if NDVI available)
            if 'NDVI_diff' in [band.getInfo().get('id') for band in change_bands]:
                vegetation_loss = change_image.select('NDVI_diff').lt(-0.2)
                classification = classification.where(vegetation_loss, 1)
            
            # Urban growth (if NDBI available)
            if 'NDBI_diff' in [band.getInfo().get('id') for band in change_bands]:
                if 'NDVI_diff' in [band.getInfo().get('id') for band in change_bands]:
                    urban_growth = change_image.select('NDBI_diff').gt(0.1).And(
                        change_image.select('NDVI_diff').lt(-0.1)
                    )
                else:
                    urban_growth = change_image.select('NDBI_diff').gt(0.1)
                classification = classification.where(urban_growth, 2)
            
            # Water change (if MNDWI available)
            if 'MNDWI_diff' in [band.getInfo().get('id') for band in change_bands]:
                water_change = change_image.select('MNDWI_diff').abs().gt(0.3)
                classification = classification.where(water_change, 3)
            
            return {
                'classification': classification.rename('land_cover_change'),
                'change_image': change_image,
                'confidence': ee.Image(1)  # Placeholder for confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying land cover changes: {str(e)}")
            raise
    
    def comprehensive_change_detection(self, before_image: ee.Image, after_image: ee.Image, 
                                     aoi: ee.Geometry, change_types: List[str] = None) -> Dict:
        """
        Perform comprehensive change detection analysis
        """
        if change_types is None:
            change_types = ['vegetation', 'urban', 'water', 'classification']
        
        results = {}
        
        try:
            # Vegetation changes
            if 'vegetation' in change_types:
                self.logger.info("Detecting vegetation changes...")
                results['vegetation'] = self.detect_vegetation_changes(
                    before_image, after_image, aoi
                )
            
            # Urban expansion
            if 'urban' in change_types:
                self.logger.info("Detecting urban expansion...")
                results['urban'] = self.detect_urban_expansion(
                    before_image, after_image, aoi
                )
            
            # Water body changes
            if 'water' in change_types:
                self.logger.info("Detecting water body changes...")
                results['water'] = self.detect_water_body_changes(
                    before_image, after_image, aoi
                )
            
            # Land cover classification
            if 'classification' in change_types:
                self.logger.info("Performing land cover change classification...")
                results['classification'] = self.classify_land_cover_change(
                    before_image, after_image, aoi
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive change detection: {str(e)}")
            raise