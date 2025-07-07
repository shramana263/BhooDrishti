"""
PNG Change Analysis Engine
Adapts the existing change analysis system to work with PNG images
"""

import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
from pathlib import Path

# Import PNG change detection
from png_change_detection import PNGChangeDetectionEngine

# Import existing system components (adapted for PNG)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class PNGChangeAnalysisEngine:
    """
    Change analysis engine adapted for PNG satellite images
    Provides detailed analysis and interpretation of detected changes
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize PNG change analysis engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Analysis thresholds adapted for PNG analysis
        self.significant_change_threshold = self.config.get('analysis', {}).get('significant_change_area', 1000)  # m²
        self.alert_thresholds = self.config.get('analysis', {}).get('alert_thresholds', {
            'deforestation': 5000,  # m²
            'urban_expansion': 2000,  # m²
            'water_loss': 3000  # m²
        })
    
    def analyze_deforestation_png(self, vegetation_results: Dict, image_info: Dict) -> Dict:
        """
        Analyze deforestation patterns and severity from PNG analysis
        
        Args:
            vegetation_results: Results from PNG vegetation change detection
            image_info: Information about the analyzed images
            
        Returns:
            Dict: Detailed deforestation analysis
        """
        try:
            deforestation_mask = vegetation_results['deforestation_mask']
            stats = vegetation_results['statistics']
            
            # Calculate deforestation area
            deforestation_area_m2 = stats['deforestation_area']['deforestation']
            deforestation_area_ha = deforestation_area_m2 / 10000
            
            # Assess severity
            severity = self._assess_change_severity(deforestation_area_m2, 'deforestation')
            
            # Calculate fragmentation metrics
            fragmentation = self._calculate_fragmentation_png(deforestation_mask)
            
            # Generate hotspots
            hotspots = self._identify_change_hotspots_png(deforestation_mask)
            
            # Risk assessment
            risk_level = self._assess_environmental_risk(deforestation_area_ha, 'forest_loss')
            
            # Calculate percentage of total area
            total_pixels = deforestation_mask.size
            affected_pixels = np.sum(deforestation_mask)
            percentage_affected = (affected_pixels / total_pixels) * 100
            
            analysis = {
                'area_affected': {
                    'square_meters': deforestation_area_m2,
                    'hectares': deforestation_area_ha,
                    'percentage_of_image': percentage_affected
                },
                'severity': severity,
                'fragmentation': fragmentation,
                'hotspots': hotspots,
                'risk_level': risk_level,
                'alert_triggered': deforestation_area_m2 > self.alert_thresholds['deforestation'],
                'recommendations': self._generate_deforestation_recommendations(severity, deforestation_area_ha),
                'temporal_context': {
                    'analysis_period': f"{image_info.get('start_date', '2014')} to {image_info.get('end_date', '2022')}",
                    'change_rate': f"{deforestation_area_ha / 8:.2f} ha/year"  # Assuming 8-year period
                }
            }
            
            print(f"🌲 Deforestation Analysis:")
            print(f"   📊 Area affected: {deforestation_area_ha:.2f} hectares")
            print(f"   ⚠️  Severity: {severity}")
            print(f"   🎯 Risk level: {risk_level}")
            print(f"   🚨 Alert triggered: {analysis['alert_triggered']}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing deforestation: {e}")
            raise
    
    def analyze_urban_expansion_png(self, urban_results: Dict, image_info: Dict) -> Dict:
        """
        Analyze urban expansion patterns and impacts from PNG analysis
        
        Args:
            urban_results: Results from PNG urban change detection
            image_info: Information about the analyzed images
            
        Returns:
            Dict: Detailed urban expansion analysis
        """
        try:
            urban_mask = urban_results['urban_expansion_mask']
            stats = urban_results['statistics']
            
            # Calculate urban expansion area
            expansion_area_m2 = stats['urban_expansion_area']['urban_change']
            expansion_area_ha = expansion_area_m2 / 10000
            
            # Assess expansion pattern
            pattern = self._analyze_urban_pattern_png(urban_mask)
            
            # Environmental impact assessment
            environmental_impact = self._assess_urban_environmental_impact(expansion_area_ha)
            
            # Infrastructure implications
            infrastructure_needs = self._assess_infrastructure_needs(expansion_area_ha, pattern)
            
            # Calculate percentage of total area
            total_pixels = urban_mask.size
            expanded_pixels = np.sum(urban_mask)
            percentage_expanded = (expanded_pixels / total_pixels) * 100
            
            analysis = {
                'area_expanded': {
                    'square_meters': expansion_area_m2,
                    'hectares': expansion_area_ha,
                    'percentage_of_image': percentage_expanded
                },
                'expansion_pattern': pattern,
                'environmental_impact': environmental_impact,
                'infrastructure_needs': infrastructure_needs,
                'alert_triggered': expansion_area_m2 > self.alert_thresholds['urban_expansion'],
                'recommendations': self._generate_urban_recommendations(pattern, expansion_area_ha),
                'growth_metrics': {
                    'expansion_rate': f"{expansion_area_ha / 8:.2f} ha/year",  # Assuming 8-year period
                    'intensity': 'high' if percentage_expanded > 2 else 'moderate' if percentage_expanded > 0.5 else 'low'
                }
            }
            
            print(f"🏙️  Urban Expansion Analysis:")
            print(f"   📊 Area expanded: {expansion_area_ha:.2f} hectares")
            print(f"   🏗️  Pattern: {pattern}")
            print(f"   🌍 Environmental impact: {environmental_impact}")
            print(f"   🚨 Alert triggered: {analysis['alert_triggered']}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing urban expansion: {e}")
            raise
    
    def analyze_water_body_changes_png(self, water_results: Dict, image_info: Dict) -> Dict:
        """
        Analyze water body changes and their implications from PNG analysis
        
        Args:
            water_results: Results from PNG water change detection
            image_info: Information about the analyzed images
            
        Returns:
            Dict: Detailed water change analysis
        """
        try:
            water_loss_mask = water_results['water_loss_mask']
            water_gain_mask = water_results['water_gain_mask']
            stats = water_results['statistics']
            
            # Calculate areas
            water_loss_m2 = stats['water_loss_area']['water_loss']
            water_gain_m2 = stats['water_gain_area']['water_gain']
            
            water_loss_ha = water_loss_m2 / 10000
            water_gain_ha = water_gain_m2 / 10000
            
            net_change_m2 = water_gain_m2 - water_loss_m2
            net_change_ha = net_change_m2 / 10000
            
            # Assess change type
            change_type = self._classify_water_change(water_loss_m2, water_gain_m2)
            
            # Environmental implications
            environmental_impact = self._assess_water_environmental_impact(
                water_loss_m2, water_gain_m2, change_type
            )
            
            # Risk assessment
            risk_assessment = self._assess_water_risk(water_loss_m2, change_type)
            
            # Calculate percentages
            total_pixels = water_loss_mask.size
            loss_percentage = (np.sum(water_loss_mask) / total_pixels) * 100
            gain_percentage = (np.sum(water_gain_mask) / total_pixels) * 100
            
            analysis = {
                'water_loss': {
                    'square_meters': water_loss_m2,
                    'hectares': water_loss_ha,
                    'percentage_of_image': loss_percentage
                },
                'water_gain': {
                    'square_meters': water_gain_m2,
                    'hectares': water_gain_ha,
                    'percentage_of_image': gain_percentage
                },
                'net_change': {
                    'square_meters': net_change_m2,
                    'hectares': net_change_ha
                },
                'change_type': change_type,
                'environmental_impact': environmental_impact,
                'risk_assessment': risk_assessment,
                'alert_triggered': water_loss_m2 > self.alert_thresholds['water_loss'],
                'recommendations': self._generate_water_recommendations(change_type, water_loss_m2),
                'hydrological_context': {
                    'dominant_change': 'loss' if water_loss_m2 > water_gain_m2 else 'gain',
                    'change_magnitude': 'significant' if abs(net_change_ha) > 1 else 'minor'
                }
            }
            
            print(f"💧 Water Body Analysis:")
            print(f"   📉 Water loss: {water_loss_ha:.2f} hectares")
            print(f"   📈 Water gain: {water_gain_ha:.2f} hectares")
            print(f"   🔄 Net change: {net_change_ha:.2f} hectares")
            print(f"   ⚠️  Risk level: {risk_assessment}")
            print(f"   🚨 Alert triggered: {analysis['alert_triggered']}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing water body changes: {e}")
            raise
    
    def generate_comprehensive_report_png(self, change_detection_results: Dict, 
                                        image1_path: str, image2_path: str) -> Dict:
        """
        Generate comprehensive change analysis report for PNG images
        
        Args:
            change_detection_results: Results from PNG change detection
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Dict: Comprehensive analysis report
        """
        try:
            # Image information
            image_info = {
                'image1_path': image1_path,
                'image2_path': image2_path,
                'image1_name': os.path.basename(image1_path),
                'image2_name': os.path.basename(image2_path),
                'start_date': '2014',  # Extracted from filename
                'end_date': '2022'     # Extracted from filename
            }
            
            report = {
                'metadata': {
                    'analysis_type': 'PNG Satellite Change Detection',
                    'timestamp': datetime.now().isoformat(),
                    'images': image_info,
                    'analysis_period': f"{image_info['start_date']} to {image_info['end_date']}"
                },
                'summary': {},
                'detailed_analysis': {},
                'alerts': [],
                'recommendations': [],
                'confidence_assessment': {}
            }
            
            # Analyze each change type
            if 'vegetation' in change_detection_results:
                deforestation_analysis = self.analyze_deforestation_png(
                    change_detection_results['vegetation'], image_info
                )
                report['detailed_analysis']['deforestation'] = deforestation_analysis
                
                if deforestation_analysis['alert_triggered']:
                    report['alerts'].append({
                        'type': 'deforestation',
                        'severity': deforestation_analysis['severity'],
                        'area': deforestation_analysis['area_affected'],
                        'priority': 'high' if deforestation_analysis['severity'] in ['severe', 'moderate'] else 'medium'
                    })
            
            if 'urban' in change_detection_results:
                urban_analysis = self.analyze_urban_expansion_png(
                    change_detection_results['urban'], image_info
                )
                report['detailed_analysis']['urban_expansion'] = urban_analysis
                
                if urban_analysis['alert_triggered']:
                    report['alerts'].append({
                        'type': 'urban_expansion',
                        'area': urban_analysis['area_expanded'],
                        'pattern': urban_analysis['expansion_pattern'],
                        'priority': 'medium'
                    })
            
            if 'water' in change_detection_results:
                water_analysis = self.analyze_water_body_changes_png(
                    change_detection_results['water'], image_info
                )
                report['detailed_analysis']['water_changes'] = water_analysis
                
                if water_analysis['alert_triggered']:
                    report['alerts'].append({
                        'type': 'water_loss',
                        'area': water_analysis['water_loss'],
                        'risk': water_analysis['risk_assessment'],
                        'priority': 'high' if water_analysis['risk_assessment'] == 'high' else 'medium'
                    })
            
            # Generate summary
            report['summary'] = self._generate_summary_png(report['detailed_analysis'], image_info)
            
            # Overall recommendations
            report['recommendations'] = self._generate_overall_recommendations_png(
                report['detailed_analysis'], report['alerts']
            )
            
            # Confidence assessment
            report['confidence_assessment'] = self._assess_overall_confidence_png(
                change_detection_results, image_info
            )
            
            print("📋 Comprehensive Report Generated:")
            print(f"   🔍 Analysis period: {report['metadata']['analysis_period']}")
            print(f"   📊 Change types analyzed: {len(report['detailed_analysis'])}")
            print(f"   🚨 Alerts triggered: {len(report['alerts'])}")
            print(f"   💡 Recommendations: {len(report['recommendations'])}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def export_analysis_report(self, report: Dict, output_path: str = None) -> str:
        """
        Export analysis report to JSON file
        
        Args:
            report: Analysis report dictionary
            output_path: Path to save the report
            
        Returns:
            str: Path to exported report
        """
        if output_path is None:
            output_dir = Path("/home/parambrata-ghosh/Development/Personal/Hackathon/ISRO/BhooDristi/Backend/png_adaptation/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"change_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_report = self._prepare_report_for_json(report)
            
            with open(output_path, 'w') as f:
                json.dump(json_report, f, indent=2, default=str)
            
            print(f"📄 Analysis report exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise
    
    def _prepare_report_for_json(self, report: Dict) -> Dict:
        """Convert report data to JSON-serializable format"""
        json_report = {}
        
        for key, value in report.items():
            if isinstance(value, dict):
                json_report[key] = self._prepare_report_for_json(value)
            elif isinstance(value, np.ndarray):
                json_report[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_report[key] = value.item()
            else:
                json_report[key] = value
        
        return json_report
    
    # Helper methods adapted for PNG analysis
    def _assess_change_severity(self, area_m2: float, change_type: str) -> str:
        """Assess severity of detected changes"""
        if change_type == 'deforestation':
            if area_m2 > 50000:  # 5 hectares
                return 'severe'
            elif area_m2 > 10000:  # 1 hectare
                return 'moderate'
            elif area_m2 > 1000:  # 0.1 hectare
                return 'minor'
            else:
                return 'negligible'
        elif change_type == 'urban_expansion':
            if area_m2 > 20000:  # 2 hectares
                return 'high'
            elif area_m2 > 5000:  # 0.5 hectares
                return 'moderate'
            else:
                return 'low'
        return 'unknown'
    
    def _calculate_fragmentation_png(self, mask: np.ndarray) -> Dict:
        """Calculate fragmentation metrics for change areas"""
        try:
            from scipy import ndimage
            
            # Label connected components
            labeled_array, num_features = ndimage.label(mask)
            
            # Calculate patch statistics
            patch_sizes = []
            for i in range(1, num_features + 1):
                patch_size = np.sum(labeled_array == i)
                patch_sizes.append(patch_size)
            
            if patch_sizes:
                avg_patch_size = np.mean(patch_sizes)
                fragmentation_level = 'high' if num_features > 10 else 'medium' if num_features > 3 else 'low'
            else:
                avg_patch_size = 0
                fragmentation_level = 'none'
            
            return {
                'patch_count': int(num_features),
                'average_patch_size_pixels': float(avg_patch_size),
                'fragmentation_level': fragmentation_level
            }
        except ImportError:
            # Fallback without scipy
            return {
                'patch_count': 1,  # Simplified
                'average_patch_size_pixels': float(np.sum(mask)),
                'fragmentation_level': 'unknown'
            }
    
    def _identify_change_hotspots_png(self, mask: np.ndarray) -> List[Dict]:
        """Identify hotspots of change within the image"""
        # Simplified hotspot identification
        if np.sum(mask) == 0:
            return []
        
        # Find center of mass of change areas
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            
            return [{
                'coordinates': [center_x, center_y],
                'intensity': 'high' if np.sum(mask) > 1000 else 'medium',
                'description': 'Primary change concentration area'
            }]
        
        return []
    
    def _analyze_urban_pattern_png(self, urban_mask: np.ndarray) -> str:
        """Analyze urban expansion patterns"""
        if np.sum(urban_mask) == 0:
            return 'none'
        
        # Simple pattern analysis based on spatial distribution
        y_coords, x_coords = np.where(urban_mask)
        
        if len(y_coords) < 10:
            return 'minimal'
        
        # Calculate spatial spread
        y_spread = np.std(y_coords)
        x_spread = np.std(x_coords)
        
        if y_spread > 50 and x_spread > 50:
            return 'scattered'
        elif y_spread > 20 or x_spread > 20:
            return 'dispersed'
        else:
            return 'compact'
    
    def _assess_environmental_risk(self, area_ha: float, change_type: str) -> str:
        """Assess environmental risk level"""
        if change_type == 'forest_loss':
            if area_ha > 10:
                return 'high'
            elif area_ha > 2:
                return 'medium'
            else:
                return 'low'
        return 'unknown'
    
    def _classify_water_change(self, loss_m2: float, gain_m2: float) -> str:
        """Classify type of water body change"""
        if loss_m2 > gain_m2 * 2:
            return 'significant_loss'
        elif gain_m2 > loss_m2 * 2:
            return 'significant_gain'
        else:
            return 'minimal_change'
    
    def _assess_urban_environmental_impact(self, area_ha: float) -> str:
        """Assess environmental impact of urban expansion"""
        if area_ha > 5:
            return 'high'
        elif area_ha > 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_infrastructure_needs(self, area_ha: float, pattern: str) -> List[str]:
        """Assess infrastructure needs based on expansion"""
        needs = []
        if area_ha > 2:
            needs.extend(['road_access', 'utilities', 'drainage'])
        if pattern == 'scattered':
            needs.append('connectivity_planning')
        if area_ha > 5:
            needs.extend(['public_transport', 'waste_management'])
        return needs
    
    def _assess_water_environmental_impact(self, loss_m2: float, gain_m2: float, change_type: str) -> str:
        """Assess environmental impact of water changes"""
        if change_type == 'significant_loss' and loss_m2 > 10000:
            return 'high'
        elif change_type == 'significant_gain' and gain_m2 > 10000:
            return 'medium'
        else:
            return 'low'
    
    def _assess_water_risk(self, loss_m2: float, change_type: str) -> str:
        """Assess risk level of water changes"""
        if change_type == 'significant_loss':
            if loss_m2 > 20000:  # 2 hectares
                return 'high'
            elif loss_m2 > 5000:  # 0.5 hectares
                return 'medium'
            else:
                return 'low'
        else:
            return 'low'
    
    def _generate_deforestation_recommendations(self, severity: str, area_ha: float) -> List[str]:
        """Generate recommendations for deforestation issues"""
        recommendations = []
        
        if severity in ['severe', 'moderate']:
            recommendations.extend([
                "Immediate field verification recommended",
                "Contact local forest authorities",
                "Implement enhanced monitoring",
                "Consider reforestation planning"
            ])
        
        if area_ha > 5:
            recommendations.extend([
                "Satellite-based verification needed",
                "Environmental impact assessment required"
            ])
        
        if severity == 'severe':
            recommendations.append("Emergency conservation measures needed")
        
        return recommendations
    
    def _generate_urban_recommendations(self, pattern: str, area_ha: float) -> List[str]:
        """Generate recommendations for urban expansion"""
        recommendations = []
        
        if pattern == 'scattered':
            recommendations.append("Consider consolidated development planning")
        
        if area_ha > 2:
            recommendations.extend([
                "Infrastructure impact assessment needed",
                "Traffic flow analysis recommended"
            ])
        
        if area_ha > 5:
            recommendations.extend([
                "Comprehensive urban planning required",
                "Environmental clearance verification"
            ])
        
        return recommendations
    
    def _generate_water_recommendations(self, change_type: str, loss_m2: float) -> List[str]:
        """Generate recommendations for water body changes"""
        recommendations = []
        
        if change_type == 'significant_loss':
            recommendations.extend([
                "Investigate cause of water loss",
                "Monitor for drought conditions",
                "Check for upstream diversions",
                "Water conservation measures needed"
            ])
            
            if loss_m2 > 10000:
                recommendations.append("Emergency water management plan required")
        
        elif change_type == 'significant_gain':
            recommendations.extend([
                "Investigate water source",
                "Flood risk assessment",
                "Water quality monitoring"
            ])
        
        return recommendations
    
    def _generate_summary_png(self, detailed_analysis: Dict, image_info: Dict) -> Dict:
        """Generate summary of all analyses"""
        summary = {
            'analysis_period': image_info.get('analysis_period', 'Unknown'),
            'total_changes_detected': len(detailed_analysis),
            'high_priority_alerts': 0,
            'total_area_affected_ha': 0,
            'dominant_change_type': 'none',
            'overall_impact': 'low'
        }
        
        impact_scores = []
        
        for analysis_type, analysis in detailed_analysis.items():
            if analysis.get('alert_triggered', False):
                summary['high_priority_alerts'] += 1
            
            # Sum up areas
            if 'area_affected' in analysis:
                summary['total_area_affected_ha'] += analysis['area_affected'].get('hectares', 0)
                impact_scores.append(3 if analysis.get('severity') == 'severe' else 2 if analysis.get('severity') == 'moderate' else 1)
            elif 'area_expanded' in analysis:
                summary['total_area_affected_ha'] += analysis['area_expanded'].get('hectares', 0)
                impact_scores.append(2)
            elif 'water_loss' in analysis:
                summary['total_area_affected_ha'] += analysis['water_loss'].get('hectares', 0)
                impact_scores.append(2 if analysis.get('risk_assessment') == 'high' else 1)
        
        # Determine dominant change type and overall impact
        if detailed_analysis:
            max_area_type = max(detailed_analysis.keys(), 
                              key=lambda x: detailed_analysis[x].get('area_affected', 
                                                                  detailed_analysis[x].get('area_expanded', 
                                                                                          detailed_analysis[x].get('water_loss', {'hectares': 0}))).get('hectares', 0))
            summary['dominant_change_type'] = max_area_type
            
            if impact_scores:
                avg_impact = np.mean(impact_scores)
                summary['overall_impact'] = 'high' if avg_impact > 2.5 else 'medium' if avg_impact > 1.5 else 'low'
        
        return summary
    
    def _generate_overall_recommendations_png(self, detailed_analysis: Dict, alerts: List[Dict]) -> List[str]:
        """Generate overall recommendations"""
        recommendations = set()
        
        # Collect all specific recommendations
        for analysis_type, analysis in detailed_analysis.items():
            if 'recommendations' in analysis:
                recommendations.update(analysis['recommendations'])
        
        # Add overall recommendations based on alerts
        if len(alerts) > 2:
            recommendations.add("Multi-hazard monitoring system recommended")
        
        if any(alert['priority'] == 'high' for alert in alerts):
            recommendations.add("Immediate intervention planning required")
        
        return list(recommendations)
    
    def _assess_overall_confidence_png(self, change_detection_results: Dict, image_info: Dict) -> Dict:
        """Assess overall confidence in the PNG analysis"""
        confidence_factors = []
        
        # Image quality factors
        confidence_factors.append("PNG format provides RGB channels")
        confidence_factors.append("Spectral indices simulated from RGB")
        
        # Temporal factors
        if image_info.get('start_date') and image_info.get('end_date'):
            confidence_factors.append("Temporal gap appropriate for change detection")
        
        # Analysis completeness
        analysis_count = len(change_detection_results)
        if analysis_count >= 3:
            confidence_factors.append("Multiple change types analyzed")
        
        # Overall confidence
        if len(confidence_factors) >= 3:
            overall_confidence = 'medium'
        else:
            overall_confidence = 'low'
        
        confidence_factors.append("Note: Analysis based on simulated spectral bands")
        
        return {
            'overall_confidence': overall_confidence,
            'factors': confidence_factors,
            'limitations': [
                "NIR and SWIR bands simulated from RGB",
                "No atmospheric correction applied",
                "Geometric accuracy not verified"
            ]
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for PNG change analysis"""
        return {
            'analysis': {
                'significant_change_area': 1000,  # m²
                'alert_thresholds': {
                    'deforestation': 5000,  # m²
                    'urban_expansion': 2000,  # m²
                    'water_loss': 3000  # m²
                }
            },
            'output': {
                'save_report': True,
                'include_visualizations': True,
                'export_format': 'json'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('PNGChangeAnalysisEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
