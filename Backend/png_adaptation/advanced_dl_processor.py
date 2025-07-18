"""
Advanced Deep Learning Processor for BhooDrishti
================================================

Integrates state-of-the-art deep learning models with the existing PNG adaptation system.
Provides enhanced change detection using neural networks.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime
import segmentation_models_pytorch as smp

# Import the advanced models
try:
    from advanced_dl_models import (
        AdvancedChangeDetectionSystem,
        create_advanced_system,
        preprocess_png_for_dl,
        postprocess_prediction,
        get_validation_transforms
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced models not available: {e}")
    MODELS_AVAILABLE = False

# Import existing system components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class AdvancedDLProcessor:
    """
    Advanced Deep Learning processor for satellite change detection
    """
    
    def __init__(self, config: Dict = None):
        if not MODELS_AVAILABLE:
            raise ImportError("Advanced models not available. Please install required dependencies.")
        
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the advanced model
        self.model, self.device = create_advanced_system(
            model_type='siamese_unet', 
            device=self.device,
            image_size=self.config.get('model', {}).get('image_size', 256)
        )
        
        # Load pre-trained weights if available
        self._load_pretrained_weights()
        
        # Load additional models
        self._load_models()
        
        self.logger.info(f"Advanced DL Processor initialized on {self.device}")
    
    def _load_pretrained_weights(self):
        """Load pre-trained weights if available"""
        weights_path = self.config.get('model', {}).get('weights_path')
        if weights_path and Path(weights_path).exists():
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded pre-trained weights from {weights_path}")
            except Exception as e:
                self.logger.warning(f"Could not load weights: {e}")
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load change detection model
            self.change_model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=6,  # 2 images * 3 channels
                classes=1,
                activation=None
            )
            
            # Ensure model is in float32
            self.change_model = self.change_model.float()
            self.change_model.to(self.device)
            self.change_model.eval()
            
            # Load other models similarly
            self.cloud_model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet", 
                in_channels=3,
                classes=3,  # clear, cloud, shadow
                activation=None
            ).float().to(self.device)
            
            self.logger.info("Loaded pre-trained models: Change and Cloud models")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def detect_changes_advanced(self, image1_path: str, image2_path: str) -> Dict:
        """
        Advanced change detection using deep learning
        """
        try:
            # Load and preprocess images
            img1_original = cv2.imread(image1_path)
            img2_original = cv2.imread(image2_path)
            
            if img1_original is None or img2_original is None:
                raise ValueError("Could not load one or both images")
            
            original_size = (img1_original.shape[1], img1_original.shape[0])
            
            # Preprocess for deep learning
            img1_tensor = preprocess_png_for_dl(image1_path, 
                                               target_size=(self.config['model']['image_size'], 
                                                           self.config['model']['image_size']))
            img2_tensor = preprocess_png_for_dl(image2_path,
                                               target_size=(self.config['model']['image_size'], 
                                                           self.config['model']['image_size']))
            
            # Move to device
            img1_tensor = img1_tensor.to(self.device)
            img2_tensor = img2_tensor.to(self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(img1_tensor, img2_tensor)
            
            # Post-process results
            change_map, change_prob = postprocess_prediction(outputs['change_map'], original_size)
            raw_change_map, _ = postprocess_prediction(outputs['raw_change_map'], original_size)
            
            # Process cloud masks
            cloud_mask1 = self._process_cloud_mask(outputs['cloud_mask1'], original_size)
            cloud_mask2 = self._process_cloud_mask(outputs['cloud_mask2'], original_size)
            
            # Extract quality scores
            quality1 = float(outputs['quality1'].cpu().numpy())
            quality2 = float(outputs['quality2'].cpu().numpy())
            
            # Calculate statistics
            total_pixels = change_map.shape[0] * change_map.shape[1]
            changed_pixels = np.sum(change_map > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Calculate change area (assuming 1 pixel = 1 m²)
            pixel_area_m2 = self.config.get('analysis', {}).get('pixel_area_m2', 1.0)
            change_area_m2 = changed_pixels * pixel_area_m2
            change_area_ha = change_area_m2 / 10000
            
            return {
                'change_map': change_map,
                'change_probability': change_prob,
                'raw_change_map': raw_change_map,
                'cloud_mask1': cloud_mask1,
                'cloud_mask2': cloud_mask2,
                'quality_scores': {
                    'image1_quality': quality1,
                    'image2_quality': quality2,
                    'average_quality': (quality1 + quality2) / 2
                },
                'statistics': {
                    'total_pixels': total_pixels,
                    'changed_pixels': changed_pixels,
                    'change_percentage': change_percentage,
                    'change_area_m2': change_area_m2,
                    'change_area_ha': change_area_ha
                },
                'metadata': {
                    'model_type': 'advanced_siamese_unet',
                    'device_used': str(self.device),
                    'image_size': self.config['model']['image_size'],
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced change detection: {e}")
            raise
    
    def _process_cloud_mask(self, cloud_mask_tensor, original_size):
        """Process cloud mask tensor to numpy array"""
        # Get the cloud probability (class 1)
        cloud_prob = cloud_mask_tensor[0, 1].cpu().numpy()
        cloud_mask_resized = cv2.resize(cloud_prob, original_size, interpolation=cv2.INTER_LINEAR)
        return (cloud_mask_resized > 0.5).astype(np.uint8) * 255
    
    def analyze_change_types_advanced(self, change_results: Dict, image1_path: str, image2_path: str) -> Dict:
        """
        Advanced analysis of detected changes
        """
        try:
            change_map = change_results['change_map']
            change_prob = change_results['change_probability']
            
            # Load original images for additional analysis
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Calculate spectral indices for change type classification
            ndvi1 = self._calculate_mock_ndvi(img1_rgb)
            ndvi2 = self._calculate_mock_ndvi(img2_rgb)
            ndvi_diff = ndvi2 - ndvi1
            
            # Classify change types based on change map and spectral information
            vegetation_change = self._classify_vegetation_change(change_map, ndvi_diff)
            urban_change = self._classify_urban_change(change_map, img1_rgb, img2_rgb)
            water_change = self._classify_water_change(change_map, img1_rgb, img2_rgb)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                change_prob, 
                change_results['quality_scores']['average_quality'],
                change_results['cloud_mask1'],
                change_results['cloud_mask2']
            )
            
            return {
                'vegetation_changes': vegetation_change,
                'urban_changes': urban_change,
                'water_changes': water_change,
                'confidence_scores': confidence_scores,
                'overall_assessment': self._generate_overall_assessment(
                    vegetation_change, urban_change, water_change, confidence_scores
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced change type analysis: {e}")
            raise
    
    def _calculate_mock_ndvi(self, rgb_image):
        """Calculate mock NDVI from RGB image"""
        # Simple approximation: use red and green channels
        red = rgb_image[:, :, 0].astype(np.float32)
        green = rgb_image[:, :, 1].astype(np.float32)
        
        # Avoid division by zero
        denominator = red + green + 1e-8
        ndvi = (green - red) / denominator
        
        return np.clip(ndvi, -1, 1)
    
    def _classify_vegetation_change(self, change_map, ndvi_diff):
        """Classify vegetation changes"""
        vegetation_mask = change_map > 0
        vegetation_ndvi_change = ndvi_diff[vegetation_mask]
        
        if len(vegetation_ndvi_change) == 0:
            return {
                'area_ha': 0,
                'type': 'no_change',
                'severity': 'none',
                'description': 'No vegetation changes detected'
            }
        
        mean_ndvi_change = np.mean(vegetation_ndvi_change)
        area_pixels = np.sum(vegetation_mask)
        area_ha = area_pixels / 10000  # Assuming 1 pixel = 1 m²
        
        if mean_ndvi_change < -0.1:
            change_type = 'deforestation'
            severity = 'high' if mean_ndvi_change < -0.3 else 'moderate'
        elif mean_ndvi_change > 0.1:
            change_type = 'vegetation_growth'
            severity = 'high' if mean_ndvi_change > 0.3 else 'moderate'
        else:
            change_type = 'vegetation_stress'
            severity = 'low'
        
        return {
            'area_ha': area_ha,
            'type': change_type,
            'severity': severity,
            'mean_ndvi_change': mean_ndvi_change,
            'description': f'{change_type.replace("_", " ").title()} detected over {area_ha:.2f} hectares'
        }
    
    def _classify_urban_change(self, change_map, img1, img2):
        """Classify urban expansion changes"""
        # Simple heuristic: urban areas tend to be less green and more uniform
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Calculate brightness increase (potential urbanization)
        brightness_diff = gray2.astype(np.float32) - gray1.astype(np.float32)
        urban_mask = (change_map > 0) & (brightness_diff > 20)
        
        area_pixels = np.sum(urban_mask)
        area_ha = area_pixels / 10000
        
        if area_ha > 0.1:  # Minimum threshold for urban change
            return {
                'area_ha': area_ha,
                'type': 'urban_expansion',
                'severity': 'high' if area_ha > 5 else 'moderate',
                'description': f'Urban expansion detected over {area_ha:.2f} hectares'
            }
        else:
            return {
                'area_ha': 0,
                'type': 'no_change',
                'severity': 'none',
                'description': 'No significant urban changes detected'
            }
    
    def _classify_water_change(self, change_map, img1, img2):
        """Classify water body changes"""
        # Simple heuristic: water bodies are typically blue and dark
        blue1 = img1[:, :, 2].astype(np.float32)
        blue2 = img2[:, :, 2].astype(np.float32)
        
        # Calculate blue channel change
        blue_diff = blue2 - blue1
        water_mask = (change_map > 0) & (np.abs(blue_diff) > 30)
        
        area_pixels = np.sum(water_mask)
        area_ha = area_pixels / 10000
        
        if area_ha > 0.05:  # Minimum threshold for water change
            mean_blue_change = np.mean(blue_diff[water_mask])
            change_type = 'water_loss' if mean_blue_change < 0 else 'water_gain'
            
            return {
                'area_ha': area_ha,
                'type': change_type,
                'severity': 'high' if area_ha > 2 else 'moderate',
                'mean_blue_change': mean_blue_change,
                'description': f'{change_type.replace("_", " ").title()} detected over {area_ha:.2f} hectares'
            }
        else:
            return {
                'area_ha': 0,
                'type': 'no_change',
                'severity': 'none',
                'description': 'No significant water body changes detected'
            }
    
    def _calculate_confidence_scores(self, change_prob, avg_quality, cloud_mask1, cloud_mask2):
        """Calculate confidence scores for the analysis"""
        # Base confidence from model probability
        mean_change_prob = np.mean(change_prob[change_prob > 0.5]) if np.any(change_prob > 0.5) else 0
        
        # Quality factor
        quality_factor = avg_quality
        
        # Cloud coverage factor
        cloud_coverage1 = np.sum(cloud_mask1 > 0) / cloud_mask1.size
        cloud_coverage2 = np.sum(cloud_mask2 > 0) / cloud_mask2.size
        avg_cloud_coverage = (cloud_coverage1 + cloud_coverage2) / 2
        cloud_factor = 1 - avg_cloud_coverage
        
        # Combined confidence
        overall_confidence = mean_change_prob * quality_factor * cloud_factor
        
        return {
            'model_confidence': float(mean_change_prob),
            'quality_factor': float(quality_factor),
            'cloud_factor': float(cloud_factor),
            'overall_confidence': float(overall_confidence),
            'cloud_coverage_percent': float(avg_cloud_coverage * 100)
        }
    
    def _generate_overall_assessment(self, vegetation_change, urban_change, water_change, confidence_scores):
        """Generate overall assessment of changes"""
        total_area = vegetation_change['area_ha'] + urban_change['area_ha'] + water_change['area_ha']
        
        dominant_change = 'none'
        max_area = 0
        
        for change_type, change_data in [
            ('vegetation', vegetation_change),
            ('urban', urban_change),
            ('water', water_change)
        ]:
            if change_data['area_ha'] > max_area:
                max_area = change_data['area_ha']
                dominant_change = change_type
        
        # Determine overall impact level
        if total_area > 10:
            impact_level = 'high'
        elif total_area > 2:
            impact_level = 'moderate'
        elif total_area > 0.1:
            impact_level = 'low'
        else:
            impact_level = 'minimal'
        
        # Generate alerts
        alerts = []
        if vegetation_change['type'] == 'deforestation' and vegetation_change['area_ha'] > 5:
            alerts.append({
                'type': 'environmental',
                'severity': 'high',
                'message': f"Significant deforestation detected: {vegetation_change['area_ha']:.2f} hectares"
            })
        
        if urban_change['area_ha'] > 3:
            alerts.append({
                'type': 'development',
                'severity': 'moderate',
                'message': f"Large urban expansion detected: {urban_change['area_ha']:.2f} hectares"
            })
        
        if water_change['type'] == 'water_loss' and water_change['area_ha'] > 1:
            alerts.append({
                'type': 'environmental',
                'severity': 'moderate',
                'message': f"Water body loss detected: {water_change['area_ha']:.2f} hectares"
            })
        
        return {
            'total_change_area_ha': total_area,
            'dominant_change_type': dominant_change,
            'impact_level': impact_level,
            'overall_confidence': confidence_scores['overall_confidence'],
            'alerts': alerts,
            'summary': f"Detected {impact_level} impact changes over {total_area:.2f} hectares, "
                      f"primarily {dominant_change} changes with {confidence_scores['overall_confidence']:.2f} confidence"
        }
    
    def create_advanced_visualization(self, change_results: Dict, analysis_results: Dict, 
                                    image1_path: str, image2_path: str, output_dir: str) -> Dict:
        """
        Create comprehensive visualization of advanced analysis results
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load original images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Create comprehensive dashboard
            fig = plt.figure(figsize=(20, 16))
            
            # Original images
            ax1 = plt.subplot(3, 4, 1)
            ax1.imshow(img1_rgb)
            ax1.set_title('Original Image 1', fontweight='bold')
            ax1.axis('off')
            
            ax2 = plt.subplot(3, 4, 2)
            ax2.imshow(img2_rgb)
            ax2.set_title('Original Image 2', fontweight='bold')
            ax2.axis('off')
            
            # Change maps
            ax3 = plt.subplot(3, 4, 3)
            ax3.imshow(change_results['change_map'], cmap='Reds', alpha=0.8)
            ax3.set_title('AI-Detected Changes', fontweight='bold')
            ax3.axis('off')
            
            ax4 = plt.subplot(3, 4, 4)
            ax4.imshow(change_results['change_probability'], cmap='viridis', vmin=0, vmax=1)
            ax4.set_title('Change Probability', fontweight='bold')
            ax4.axis('off')
            
            # Cloud masks
            ax5 = plt.subplot(3, 4, 5)
            ax5.imshow(change_results['cloud_mask1'], cmap='Blues', alpha=0.7)
            ax5.set_title('Cloud Mask 1', fontweight='bold')
            ax5.axis('off')
            
            ax6 = plt.subplot(3, 4, 6)
            ax6.imshow(change_results['cloud_mask2'], cmap='Blues', alpha=0.7)
            ax6.set_title('Cloud Mask 2', fontweight='bold')
            ax6.axis('off')
            
            # Overlay visualization
            ax7 = plt.subplot(3, 4, 7)
            overlay = img2_rgb.copy()
            change_overlay = np.zeros_like(overlay)
            change_overlay[:, :, 0] = change_results['change_map']  # Red channel for changes
            overlay = cv2.addWeighted(overlay, 0.7, change_overlay, 0.3, 0)
            ax7.imshow(overlay)
            ax7.set_title('Change Overlay', fontweight='bold')
            ax7.axis('off')
            
            # Statistics and analysis text
            ax8 = plt.subplot(3, 4, (8, 12))
            ax8.axis('off')
            
            # Create detailed text summary
            stats = change_results['statistics']
            quality = change_results['quality_scores']
            overall = analysis_results['overall_assessment']
            
            summary_text = f"""
ADVANCED AI CHANGE DETECTION REPORT
{'='*50}

🤖 Model Information:
   • Architecture: {change_results['metadata']['model_type']}
   • Processing Device: {change_results['metadata']['device_used']}
   • Image Resolution: {change_results['metadata']['image_size']}px

📊 Detection Statistics:
   • Total Area Analyzed: {stats['total_pixels']:,} pixels
   • Changes Detected: {stats['changed_pixels']:,} pixels ({stats['change_percentage']:.2f}%)
   • Changed Area: {stats['change_area_ha']:.2f} hectares

🎯 Quality Assessment:
   • Image 1 Quality: {quality['image1_quality']:.3f}
   • Image 2 Quality: {quality['image2_quality']:.3f}
   • Average Quality: {quality['average_quality']:.3f}
   • Cloud Coverage: {analysis_results['confidence_scores']['cloud_coverage_percent']:.1f}%

🌍 Change Analysis:
   • Vegetation Changes: {analysis_results['vegetation_changes']['area_ha']:.2f} ha
     └─ Type: {analysis_results['vegetation_changes']['type']}
     └─ Severity: {analysis_results['vegetation_changes']['severity']}
   
   • Urban Changes: {analysis_results['urban_changes']['area_ha']:.2f} ha
     └─ Type: {analysis_results['urban_changes']['type']}
     └─ Severity: {analysis_results['urban_changes']['severity']}
   
   • Water Changes: {analysis_results['water_changes']['area_ha']:.2f} ha
     └─ Type: {analysis_results['water_changes']['type']}
     └─ Severity: {analysis_results['water_changes']['severity']}

⚠️ Overall Assessment:
   • Dominant Change: {overall['dominant_change_type']}
   • Impact Level: {overall['impact_level']}
   • Confidence: {overall['overall_confidence']:.3f}
   • Total Changed Area: {overall['total_change_area_ha']:.2f} hectares

🚨 Alerts Generated: {len(overall['alerts'])}
"""
            
            for i, alert in enumerate(overall['alerts'][:3]):  # Show max 3 alerts
                summary_text += f"   {i+1}. [{alert['severity'].upper()}] {alert['message']}\n"
            
            ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.suptitle('BhooDrishti Advanced AI Change Detection Dashboard', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save the visualization
            dashboard_path = output_dir / "advanced_ai_dashboard.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save individual change maps
            change_map_path = output_dir / "ai_change_map.png"
            cv2.imwrite(str(change_map_path), change_results['change_map'])
            
            probability_map_path = output_dir / "ai_probability_map.png"
            prob_viz = (change_results['change_probability'] * 255).astype(np.uint8)
            cv2.imwrite(str(probability_map_path), prob_viz)
            
            self.logger.info(f"Advanced visualization saved to {dashboard_path}")
            
            return {
                'dashboard_path': str(dashboard_path),
                'change_map_path': str(change_map_path),
                'probability_map_path': str(probability_map_path),
                'output_directory': str(output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating advanced visualization: {e}")
            raise
    
    def export_advanced_report(self, change_results: Dict, analysis_results: Dict, 
                             output_path: str = None) -> str:
        """Export comprehensive analysis report as JSON"""
        try:
            if output_path is None:
                output_path = f"advanced_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare comprehensive report
            report = {
                'metadata': {
                    'report_type': 'advanced_ai_change_detection',
                    'generated_at': datetime.now().isoformat(),
                    'model_info': change_results['metadata'],
                    'processor_version': '1.0.0'
                },
                'detection_results': {
                    'statistics': change_results['statistics'],
                    'quality_assessment': change_results['quality_scores'],
                    'confidence_metrics': analysis_results['confidence_scores']
                },
                'change_analysis': {
                    'vegetation_changes': analysis_results['vegetation_changes'],
                    'urban_changes': analysis_results['urban_changes'],
                    'water_changes': analysis_results['water_changes'],
                    'overall_assessment': analysis_results['overall_assessment']
                },
                'recommendations': self._generate_recommendations(analysis_results),
                'technical_details': {
                    'device_used': str(self.device),
                    'model_architecture': 'Siamese U-Net with Attention',
                    'preprocessing': 'Standard ImageNet normalization',
                    'postprocessing': 'Quality-aware confidence scaling'
                }
            }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Advanced report exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting advanced report: {e}")
            raise
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        vegetation = analysis_results['vegetation_changes']
        urban = analysis_results['urban_changes']
        water = analysis_results['water_changes']
        overall = analysis_results['overall_assessment']
        
        if vegetation['type'] == 'deforestation' and vegetation['area_ha'] > 1:
            recommendations.append({
                'category': 'environmental_protection',
                'priority': 'high',
                'action': 'Immediate forest conservation measures',
                'description': f"Deploy reforestation efforts in the {vegetation['area_ha']:.2f} hectare deforested area",
                'estimated_cost': 'High',
                'timeline': 'Immediate (0-3 months)'
            })
        
        if urban['area_ha'] > 2:
            recommendations.append({
                'category': 'urban_planning',
                'priority': 'moderate',
                'action': 'Infrastructure development planning',
                'description': f"Plan for supporting infrastructure in {urban['area_ha']:.2f} hectare urban expansion zone",
                'estimated_cost': 'Very High',
                'timeline': 'Short-term (3-12 months)'
            })
        
        if water['type'] == 'water_loss' and water['area_ha'] > 0.5:
            recommendations.append({
                'category': 'water_management',
                'priority': 'high',
                'action': 'Water conservation and restoration',
                'description': f"Investigate causes of {water['area_ha']:.2f} hectare water body loss and implement restoration",
                'estimated_cost': 'Moderate',
                'timeline': 'Short-term (1-6 months)'
            })
        
        if overall['overall_confidence'] < 0.7:
            recommendations.append({
                'category': 'monitoring',
                'priority': 'moderate',
                'action': 'Enhanced monitoring',
                'description': 'Deploy additional sensors or acquire higher quality imagery for better analysis',
                'estimated_cost': 'Moderate',
                'timeline': 'Short-term (1-3 months)'
            })
        
        return recommendations
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'model': {
                'image_size': 256,
                'weights_path': None,
                'confidence_threshold': 0.5
            },
            'analysis': {
                'pixel_area_m2': 1.0,
                'min_change_area_ha': 0.01,
                'vegetation_threshold': 0.1,
                'urban_threshold': 20,
                'water_threshold': 30
            },
            'visualization': {
                'dpi': 300,
                'figure_size': (20, 16),
                'save_individual_maps': True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('AdvancedDLProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _preprocess_images(self, image1_path: str, image2_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess images for model input"""
        try:
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None or img2 is None:
                raise ValueError("Could not load one or both images")
            
            # Convert BGR to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            target_size = (self.config['model']['image_size'], self.config['model']['image_size'])
            img1 = cv2.resize(img1, target_size)
            img2 = cv2.resize(img2, target_size)
            
            # Normalize to [0, 1] and ensure float32 type
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0
            
            # Convert to tensors and ensure correct type
            tensor1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
            tensor2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
            
            # Move to device
            tensor1 = tensor1.to(self.device)
            tensor2 = tensor2.to(self.device)
            
            return tensor1, tensor2
            
        except Exception as e:
            self.logger.error(f"Error preprocessing images: {e}")
            raise

    def _detect_changes(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Detect changes using Siamese U-Net"""
        try:
            with torch.no_grad():
                # Ensure tensors are float32
                tensor1 = tensor1.float()
                tensor2 = tensor2.float()
                
                # Concatenate along channel dimension
                combined = torch.cat([tensor1, tensor2], dim=1)
                
                # Get change prediction
                change_logits = self.change_model(combined)
                change_probs = torch.sigmoid(change_logits)
                
                return change_probs
                
        except Exception as e:
            self.logger.error(f"Error in change detection: {e}")
            raise

# Main processing function for integration
def process_images_with_advanced_ai(image1_path: str, image2_path: str, 
                                   output_dir: str, config: Dict = None) -> Dict:
    """
    Main function to process images with advanced AI
    """
    try:
        # Initialize processor
        processor = AdvancedDLProcessor(config)
        
        # Detect changes
        change_results = processor.detect_changes_advanced(image1_path, image2_path)
        
        # Analyze change types
        analysis_results = processor.analyze_change_types_advanced(change_results, image1_path, image2_path)
        
        # Create visualizations
        viz_results = processor.create_advanced_visualization(
            change_results, analysis_results, image1_path, image2_path, output_dir
        )
        
        # Export report
        report_path = processor.export_advanced_report(
            change_results, analysis_results, 
            os.path.join(output_dir, "advanced_ai_report.json")
        )
        
        return {
            'success': True,
            'change_results': change_results,
            'analysis_results': analysis_results,
            'visualizations': viz_results,
            'report_path': report_path,
            'processor_info': {
                'device_used': str(processor.device),
                'model_type': 'advanced_siamese_unet'
            }
        }
        
    except Exception as e:
        logging.error(f"Error in advanced AI processing: {e}")
        return {
            'success': False,
            'error': str(e),
            'processor_info': {
                'device_used': 'unknown',
                'model_type': 'advanced_siamese_unet'
            }
        }

if __name__ == "__main__":
    # Test the advanced processor
    print("🤖 Testing Advanced AI Change Detection Processor...")
    
    # Example usage
    processor = AdvancedDLProcessor()
    print(f"✅ Processor initialized on {processor.device}")
    print("🚀 Ready for advanced AI-powered change detection!")
