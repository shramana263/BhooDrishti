"""
PNG Change Detection Engine
Adapts the existing change detection system to work with PNG images
"""

import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Import the PNG processor
from png_processor import PNGSatelliteProcessor

# Import existing system components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class PNGChangeDetectionEngine:
    """
    Change detection engine adapted for PNG satellite images
    Uses the existing system architecture but works with local PNG files
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize PNG change detection engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.png_processor = PNGSatelliteProcessor(config)
        
        # Change detection thresholds adapted for PNG analysis
        self.ndvi_threshold = self.config.get('change_detection', {}).get('ndvi_threshold', 0.1)
        self.confidence_threshold = self.config.get('change_detection', {}).get('confidence_threshold', 0.5)
        self.urban_threshold = self.config.get('change_detection', {}).get('urban_threshold', 0.05)
        self.water_threshold = self.config.get('change_detection', {}).get('water_threshold', 0.3)
    
    def detect_vegetation_changes_png(self, image1_path: str, image2_path: str) -> Dict:
        """
        Detect vegetation changes between two PNG images using NDVI analysis
        
        Args:
            image1_path: Path to first (earlier) PNG image
            image2_path: Path to second (later) PNG image
            
        Returns:
            Dict: Vegetation change analysis results
        """
        try:
            print("🌱 Detecting vegetation changes from PNG images...")
            
            # Load and process PNG images
            img1_data = self.png_processor.load_png_image(image1_path)
            img2_data = self.png_processor.load_png_image(image2_path)
            
            if not img1_data or not img2_data:
                raise ValueError("Failed to load PNG images")
            
            # Calculate spectral indices
            indices1 = self.png_processor.calculate_spectral_indices(
                img1_data['data'], img1_data['metadata']['band_names']
            )
            indices2 = self.png_processor.calculate_spectral_indices(
                img2_data['data'], img2_data['metadata']['band_names']
            )
            
            if 'ndvi' not in indices1 or 'ndvi' not in indices2:
                raise ValueError("NDVI calculation failed")
            
            # Calculate NDVI difference
            ndvi1 = indices1['ndvi']
            ndvi2 = indices2['ndvi']
            ndvi_diff = ndvi2 - ndvi1
            
            # Create change masks
            deforestation_mask = ndvi_diff < -self.ndvi_threshold  # Vegetation loss
            afforestation_mask = ndvi_diff > self.ndvi_threshold   # Vegetation gain
            
            # Calculate statistics
            pixel_area = 100  # Assumed 10m x 10m pixels = 100 m² per pixel
            
            deforestation_pixels = np.sum(deforestation_mask)
            afforestation_pixels = np.sum(afforestation_mask)
            
            deforestation_area_m2 = deforestation_pixels * pixel_area
            afforestation_area_m2 = afforestation_pixels * pixel_area
            
            mean_ndvi_change = np.mean(ndvi_diff)
            
            # Calculate statistics
            stats = {
                'deforestation_area': {
                    'deforestation': deforestation_area_m2
                },
                'afforestation_area': {
                    'afforestation': afforestation_area_m2
                },
                'mean_ndvi_change': {
                    'ndvi_diff': mean_ndvi_change
                }
            }
            
            print(f"📊 Vegetation Analysis Results:")
            print(f"   🌲 Deforestation: {deforestation_area_m2:.0f} m² ({deforestation_area_m2/10000:.3f} ha)")
            print(f"   🌱 Afforestation: {afforestation_area_m2:.0f} m² ({afforestation_area_m2/10000:.3f} ha)")
            print(f"   📈 Mean NDVI change: {mean_ndvi_change:.3f}")
            
            return {
                'change_image': ndvi_diff,
                'deforestation_mask': deforestation_mask,
                'afforestation_mask': afforestation_mask,
                'statistics': stats,
                'ndvi_before': ndvi1,
                'ndvi_after': ndvi2
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting vegetation changes: {e}")
            raise
    
    def detect_urban_expansion_png(self, image1_path: str, image2_path: str) -> Dict:
        """
        Detect urban expansion between two PNG images
        
        Args:
            image1_path: Path to first (earlier) PNG image
            image2_path: Path to second (later) PNG image
            
        Returns:
            Dict: Urban expansion analysis results
        """
        try:
            print("🏢 Detecting urban expansion from PNG images...")
            
            # Load and process PNG images
            img1_data = self.png_processor.load_png_image(image1_path)
            img2_data = self.png_processor.load_png_image(image2_path)
            
            if not img1_data or not img2_data:
                raise ValueError("Failed to load PNG images")
            
            # Calculate spectral indices
            indices1 = self.png_processor.calculate_spectral_indices(
                img1_data['data'], img1_data['metadata']['band_names']
            )
            indices2 = self.png_processor.calculate_spectral_indices(
                img2_data['data'], img2_data['metadata']['band_names']
            )
            
            # Use NDBI for urban detection if available, else use brightness
            if 'ndbi' in indices1 and 'ndbi' in indices2:
                ndbi1 = indices1['ndbi']
                ndbi2 = indices2['ndbi']
                ndbi_diff = ndbi2 - ndbi1
                
                # Urban expansion mask
                urban_expansion = ndbi_diff > self.urban_threshold
                
                # Additional criteria: NDVI decrease
                if 'ndvi' in indices1 and 'ndvi' in indices2:
                    ndvi_decrease = (indices1['ndvi'] - indices2['ndvi']) > 0.1
                    urban_change = np.logical_and(urban_expansion, ndvi_decrease)
                else:
                    urban_change = urban_expansion
                
            else:
                # Fallback: use brightness increase
                brightness1 = np.mean(img1_data['data'][:3], axis=0)  # RGB mean
                brightness2 = np.mean(img2_data['data'][:3], axis=0)  # RGB mean
                brightness_increase = brightness2 - brightness1
                urban_change = brightness_increase > 0.05
                ndbi_diff = brightness_increase  # Use brightness as proxy
            
            # Calculate statistics
            pixel_area = 100  # Assumed 10m x 10m pixels
            urban_pixels = np.sum(urban_change)
            urban_area_m2 = urban_pixels * pixel_area
            
            mean_ndbi_change = np.mean(ndbi_diff)
            
            stats = {
                'urban_expansion_area': {
                    'urban_change': urban_area_m2
                },
                'mean_ndbi_change': {
                    'NDBI_diff': mean_ndbi_change
                }
            }
            
            print(f"📊 Urban Expansion Results:")
            print(f"   🏙️  Urban expansion: {urban_area_m2:.0f} m² ({urban_area_m2/10000:.3f} ha)")
            print(f"   📈 Mean NDBI change: {mean_ndbi_change:.3f}")
            
            return {
                'change_image': ndbi_diff,
                'urban_expansion_mask': urban_change,
                'statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting urban expansion: {e}")
            raise
    
    def detect_water_body_changes_png(self, image1_path: str, image2_path: str) -> Dict:
        """
        Detect water body changes between two PNG images
        
        Args:
            image1_path: Path to first (earlier) PNG image
            image2_path: Path to second (later) PNG image
            
        Returns:
            Dict: Water body change analysis results
        """
        try:
            print("💧 Detecting water body changes from PNG images...")
            
            # Load and process PNG images
            img1_data = self.png_processor.load_png_image(image1_path)
            img2_data = self.png_processor.load_png_image(image2_path)
            
            if not img1_data or not img2_data:
                raise ValueError("Failed to load PNG images")
            
            # Calculate spectral indices
            indices1 = self.png_processor.calculate_spectral_indices(
                img1_data['data'], img1_data['metadata']['band_names']
            )
            indices2 = self.png_processor.calculate_spectral_indices(
                img2_data['data'], img2_data['metadata']['band_names']
            )
            
            # Use MNDWI for water detection
            if 'mndwi' in indices1 and 'mndwi' in indices2:
                mndwi1 = indices1['mndwi']
                mndwi2 = indices2['mndwi']
            else:
                # Fallback: simple blue channel analysis
                blue1 = img1_data['data'][2]  # Blue channel
                blue2 = img2_data['data'][2]  # Blue channel
                green1 = img1_data['data'][1]  # Green channel
                green2 = img2_data['data'][1]  # Green channel
                
                # Simple water index (higher blue relative to green)
                mndwi1 = (blue1 - green1) / (blue1 + green1 + 1e-8)
                mndwi2 = (blue2 - green2) / (blue2 + green2 + 1e-8)
            
            # Water masks
            water_threshold = 0.0  # MNDWI > 0 typically indicates water
            water_mask1 = mndwi1 > water_threshold
            water_mask2 = mndwi2 > water_threshold
            
            # Calculate changes
            water_loss = np.logical_and(water_mask1, ~water_mask2)  # Was water, now not
            water_gain = np.logical_and(~water_mask1, water_mask2)  # Was not water, now is
            
            # Calculate statistics
            pixel_area = 100  # Assumed 10m x 10m pixels
            
            water_loss_pixels = np.sum(water_loss)
            water_gain_pixels = np.sum(water_gain)
            
            water_loss_m2 = water_loss_pixels * pixel_area
            water_gain_m2 = water_gain_pixels * pixel_area
            
            stats = {
                'water_loss_area': {
                    'water_loss': water_loss_m2
                },
                'water_gain_area': {
                    'water_gain': water_gain_m2
                }
            }
            
            print(f"📊 Water Body Analysis Results:")
            print(f"   💧 Water loss: {water_loss_m2:.0f} m² ({water_loss_m2/10000:.3f} ha)")
            print(f"   🌊 Water gain: {water_gain_m2:.0f} m² ({water_gain_m2/10000:.3f} ha)")
            
            return {
                'water_loss_mask': water_loss,
                'water_gain_mask': water_gain,
                'before_water_mask': water_mask1,
                'after_water_mask': water_mask2,
                'statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting water body changes: {e}")
            raise
    
    def comprehensive_change_detection_png(self, image1_path: str, image2_path: str,
                                         change_types: List[str] = None) -> Dict:
        """
        Perform comprehensive change detection on PNG images
        
        Args:
            image1_path: Path to first (earlier) PNG image
            image2_path: Path to second (later) PNG image
            change_types: List of change types to detect
            
        Returns:
            Dict: Comprehensive change detection results
        """
        if change_types is None:
            change_types = ['vegetation', 'urban', 'water']
        
        results = {}
        
        try:
            print("🔍 Performing comprehensive change detection on PNG images...")
            print(f"📂 Image 1: {os.path.basename(image1_path)}")
            print(f"📂 Image 2: {os.path.basename(image2_path)}")
            
            # Vegetation changes
            if 'vegetation' in change_types:
                results['vegetation'] = self.detect_vegetation_changes_png(image1_path, image2_path)
            
            # Urban expansion
            if 'urban' in change_types:
                results['urban'] = self.detect_urban_expansion_png(image1_path, image2_path)
            
            # Water body changes
            if 'water' in change_types:
                results['water'] = self.detect_water_body_changes_png(image1_path, image2_path)
            
            print("✅ Comprehensive change detection completed!")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive change detection: {e}")
            raise
    
    def visualize_change_results(self, results: Dict, image1_path: str, image2_path: str,
                               output_dir: str = None) -> None:
        """
        Create comprehensive visualization of change detection results
        
        Args:
            results: Change detection results
            image1_path: Path to first image
            image2_path: Path to second image
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = "/home/parambrata-ghosh/Development/Personal/Hackathon/ISRO/BhooDristi/Backend/png_adaptation/outputs"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load original images for display
        img1_data = self.png_processor.load_png_image(image1_path)
        img2_data = self.png_processor.load_png_image(image2_path)
        
        if not img1_data or not img2_data:
            print("❌ Failed to load images for visualization")
            return
        
        # Create RGB composites
        rgb1 = np.transpose(img1_data['original_rgb'], (1, 2, 0))
        rgb2 = np.transpose(img2_data['original_rgb'], (1, 2, 0))
        
        # Create comprehensive visualization
        n_plots = 2 + len(results)  # Original images + change results
        fig, axes = plt.subplots(2, max(3, n_plots//2 + 1), figsize=(20, 12))
        axes = axes.flatten()
        
        # Original images
        axes[0].imshow(rgb1)
        axes[0].set_title(f'Image 1: {os.path.basename(image1_path)}')
        axes[0].axis('off')
        
        axes[1].imshow(rgb2)
        axes[1].set_title(f'Image 2: {os.path.basename(image2_path)}')
        axes[1].axis('off')
        
        plot_idx = 2
        
        # Vegetation changes
        if 'vegetation' in results:
            veg_results = results['vegetation']
            
            # NDVI change
            if 'change_image' in veg_results:
                im = axes[plot_idx].imshow(veg_results['change_image'], cmap='RdYlGn', vmin=-0.5, vmax=0.5)
                axes[plot_idx].set_title('NDVI Change')
                axes[plot_idx].axis('off')
                plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
                plot_idx += 1
            
            # Combined vegetation changes
            if 'deforestation_mask' in veg_results and 'afforestation_mask' in veg_results:
                change_viz = np.zeros_like(veg_results['deforestation_mask'], dtype=np.float32)
                change_viz[veg_results['deforestation_mask']] = -1  # Red for deforestation
                change_viz[veg_results['afforestation_mask']] = 1   # Green for afforestation
                
                im = axes[plot_idx].imshow(change_viz, cmap='RdYlGn', vmin=-1, vmax=1)
                axes[plot_idx].set_title('Vegetation Changes\n(Red: Loss, Green: Gain)')
                axes[plot_idx].axis('off')
                plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
                plot_idx += 1
        
        # Urban expansion
        if 'urban' in results:
            urban_results = results['urban']
            
            if 'urban_expansion_mask' in urban_results:
                axes[plot_idx].imshow(urban_results['urban_expansion_mask'], cmap='Reds')
                axes[plot_idx].set_title('Urban Expansion')
                axes[plot_idx].axis('off')
                plot_idx += 1
        
        # Water changes
        if 'water' in results:
            water_results = results['water']
            
            if 'water_loss_mask' in water_results and 'water_gain_mask' in water_results:
                water_change_viz = np.zeros_like(water_results['water_loss_mask'], dtype=np.float32)
                water_change_viz[water_results['water_loss_mask']] = -1  # Red for water loss
                water_change_viz[water_results['water_gain_mask']] = 1   # Blue for water gain
                
                im = axes[plot_idx].imshow(water_change_viz, cmap='RdBu', vmin=-1, vmax=1)
                axes[plot_idx].set_title('Water Changes\n(Red: Loss, Blue: Gain)')
                axes[plot_idx].axis('off')
                plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_path / "comprehensive_change_detection.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comprehensive visualization saved: {viz_path}")
        
        # Don't show plot in non-interactive environment
        if matplotlib.get_backend() != 'Agg':
            plt.show()
        
        plt.close()  # Close figure to free memory
        
        # Save individual change maps
        self._save_individual_change_maps(results, output_path)
    
    def _save_individual_change_maps(self, results: Dict, output_path: Path) -> None:
        """Save individual change maps as separate files"""
        
        # Save vegetation changes
        if 'vegetation' in results:
            veg_results = results['vegetation']
            
            if 'change_image' in veg_results:
                plt.figure(figsize=(10, 8))
                plt.imshow(veg_results['change_image'], cmap='RdYlGn', vmin=-0.5, vmax=0.5)
                plt.title('NDVI Change (2014 to 2022)')
                plt.colorbar(label='NDVI Difference')
                plt.axis('off')
                plt.savefig(output_path / "ndvi_change.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        # Save urban expansion
        if 'urban' in results:
            urban_results = results['urban']
            
            if 'urban_expansion_mask' in urban_results:
                plt.figure(figsize=(10, 8))
                plt.imshow(urban_results['urban_expansion_mask'], cmap='Reds')
                plt.title('Urban Expansion (2014 to 2022)')
                plt.axis('off')
                plt.savefig(output_path / "urban_expansion.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"📁 Individual change maps saved to: {output_path}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for PNG change detection"""
        return {
            'change_detection': {
                'ndvi_threshold': 0.1,
                'confidence_threshold': 0.5,
                'urban_threshold': 0.05,
                'water_threshold': 0.3
            },
            'processing': {
                'enable_noise_reduction': True,
                'enable_enhancement': True
            },
            'output': {
                'save_individual_maps': True,
                'save_comprehensive_viz': True,
                'output_format': 'png'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('PNGChangeDetectionEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
