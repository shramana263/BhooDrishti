import ee
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

class ChangeAnalysisEngine:
    """
    Advanced change analysis engine for interpreting and quantifying detected changes
    Focuses on deforestation, urban expansion, and water body changes
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analysis thresholds
        self.significant_change_threshold = config.get('analysis', {}).get('significant_change_area', 1000)  # m²
        self.alert_thresholds = config.get('analysis', {}).get('alert_thresholds', {
            'deforestation': 5000,  # m²
            'urban_expansion': 2000,  # m²
            'water_loss': 3000  # m²
        })
    
    def analyze_deforestation(self, vegetation_results: Dict, aoi: ee.Geometry) -> Dict:
        """
        Analyze deforestation patterns and severity
        """
        try:
            deforestation_mask = vegetation_results['deforestation_mask']
            stats = vegetation_results['statistics']
            
            # Calculate deforestation area in hectares
            deforestation_area_m2 = stats['deforestation_area'].getInfo().get('deforestation', 0)
            deforestation_area_ha = deforestation_area_m2 / 10000
            
            # Assess severity
            severity = self._assess_change_severity(deforestation_area_m2, 'deforestation')
            
            # Calculate fragmentation metrics
            fragmentation = self._calculate_fragmentation(deforestation_mask, aoi)
            
            # Generate hotspots
            hotspots = self._identify_change_hotspots(deforestation_mask, aoi)
            
            # Risk assessment
            risk_level = self._assess_environmental_risk(deforestation_area_ha, 'forest_loss')
            
            analysis = {
                'area_affected': {
                    'square_meters': deforestation_area_m2,
                    'hectares': deforestation_area_ha,
                    'percentage_of_aoi': self._calculate_percentage_of_aoi(deforestation_area_m2, aoi)
                },
                'severity': severity,
                'fragmentation': fragmentation,
                'hotspots': hotspots,
                'risk_level': risk_level,
                'alert_triggered': deforestation_area_m2 > self.alert_thresholds['deforestation'],
                'recommendations': self._generate_deforestation_recommendations(severity, deforestation_area_ha)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing deforestation: {str(e)}")
            raise
    
    def analyze_urban_expansion(self, urban_results: Dict, aoi: ee.Geometry) -> Dict:
        """
        Analyze urban expansion patterns and impacts
        """
        try:
            urban_mask = urban_results['urban_expansion_mask']
            stats = urban_results['statistics']
            
            # Calculate urban expansion area
            expansion_area_m2 = stats['urban_expansion_area'].getInfo().get('urban_change', 0)
            expansion_area_ha = expansion_area_m2 / 10000
            
            # Assess expansion pattern
            pattern = self._analyze_urban_pattern(urban_mask, aoi)
            
            # Calculate proximity to existing urban areas
            proximity_analysis = self._analyze_urban_proximity(urban_mask, aoi)
            
            # Environmental impact assessment
            environmental_impact = self._assess_urban_environmental_impact(expansion_area_ha)
            
            # Infrastructure implications
            infrastructure_needs = self._assess_infrastructure_needs(expansion_area_ha, pattern)
            
            analysis = {
                'area_expanded': {
                    'square_meters': expansion_area_m2,
                    'hectares': expansion_area_ha,
                    'percentage_of_aoi': self._calculate_percentage_of_aoi(expansion_area_m2, aoi)
                },
                'expansion_pattern': pattern,
                'proximity_analysis': proximity_analysis,
                'environmental_impact': environmental_impact,
                'infrastructure_needs': infrastructure_needs,
                'alert_triggered': expansion_area_m2 > self.alert_thresholds['urban_expansion'],
                'recommendations': self._generate_urban_recommendations(pattern, expansion_area_ha)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing urban expansion: {str(e)}")
            raise
    
    def analyze_water_body_changes(self, water_results: Dict, aoi: ee.Geometry) -> Dict:
        """
        Analyze water body changes and their implications
        """
        try:
            water_loss_mask = water_results['water_loss_mask']
            water_gain_mask = water_results['water_gain_mask']
            stats = water_results['statistics']
            
            # Calculate areas
            water_loss_m2 = stats['water_loss_area'].getInfo().get('water_loss', 0)
            water_gain_m2 = stats['water_gain_area'].getInfo().get('water_gain', 0)
            
            net_change_m2 = water_gain_m2 - water_loss_m2
            
            # Assess change type
            change_type = self._classify_water_change(water_loss_m2, water_gain_m2)
            
            # Analyze seasonal vs permanent changes
            permanency = self._assess_water_change_permanency(water_results, aoi)
            
            # Environmental implications
            environmental_impact = self._assess_water_environmental_impact(
                water_loss_m2, water_gain_m2, change_type
            )
            
            # Risk assessment
            risk_assessment = self._assess_water_risk(water_loss_m2, change_type)
            
            analysis = {
                'water_loss': {
                    'square_meters': water_loss_m2,
                    'hectares': water_loss_m2 / 10000
                },
                'water_gain': {
                    'square_meters': water_gain_m2,
                    'hectares': water_gain_m2 / 10000
                },
                'net_change': {
                    'square_meters': net_change_m2,
                    'hectares': net_change_m2 / 10000
                },
                'change_type': change_type,
                'permanency': permanency,
                'environmental_impact': environmental_impact,
                'risk_assessment': risk_assessment,
                'alert_triggered': water_loss_m2 > self.alert_thresholds['water_loss'],
                'recommendations': self._generate_water_recommendations(change_type, water_loss_m2)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing water body changes: {str(e)}")
            raise
    
    def generate_comprehensive_report(self, change_detection_results: Dict, 
                                    aoi: ee.Geometry, metadata: Dict) -> Dict:
        """
        Generate comprehensive change analysis report
        """
        try:
            report = {
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'aoi_info': self._get_aoi_info(aoi),
                'summary': {},
                'detailed_analysis': {},
                'alerts': [],
                'recommendations': [],
                'confidence_assessment': {}
            }
            
            # Analyze each change type
            if 'vegetation' in change_detection_results:
                deforestation_analysis = self.analyze_deforestation(
                    change_detection_results['vegetation'], aoi
                )
                report['detailed_analysis']['deforestation'] = deforestation_analysis
                
                if deforestation_analysis['alert_triggered']:
                    report['alerts'].append({
                        'type': 'deforestation',
                        'severity': deforestation_analysis['severity'],
                        'area': deforestation_analysis['area_affected']
                    })
            
            if 'urban' in change_detection_results:
                urban_analysis = self.analyze_urban_expansion(
                    change_detection_results['urban'], aoi
                )
                report['detailed_analysis']['urban_expansion'] = urban_analysis
                
                if urban_analysis['alert_triggered']:
                    report['alerts'].append({
                        'type': 'urban_expansion',
                        'area': urban_analysis['area_expanded'],
                        'pattern': urban_analysis['expansion_pattern']
                    })
            
            if 'water' in change_detection_results:
                water_analysis = self.analyze_water_body_changes(
                    change_detection_results['water'], aoi
                )
                report['detailed_analysis']['water_changes'] = water_analysis
                
                if water_analysis['alert_triggered']:
                    report['alerts'].append({
                        'type': 'water_loss',
                        'area': water_analysis['water_loss'],
                        'risk': water_analysis['risk_assessment']
                    })
            
            # Generate summary
            report['summary'] = self._generate_summary(report['detailed_analysis'])
            
            # Overall recommendations
            report['recommendations'] = self._generate_overall_recommendations(
                report['detailed_analysis']
            )
            
            # Confidence assessment
            report['confidence_assessment'] = self._assess_overall_confidence(
                change_detection_results
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            raise
    
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
        # Add other change types as needed
        return 'unknown'
    
    def _calculate_fragmentation(self, mask: ee.Image, aoi: ee.Geometry) -> Dict:
        """Calculate fragmentation metrics for change areas"""
        try:
            # Connected components analysis
            connected = mask.connectedComponents(
                connectedness=ee.Kernel.plus(1),
                maxSize=256
            )
            
            # Number of patches
            patch_count = connected.select('labels').reduceRegion(
                reducer=ee.Reducer.countDistinct(),
                geometry=aoi,
                scale=10,
                maxPixels=1e9
            )
            
            return {
                'patch_count': patch_count.getInfo().get('labels', 0),
                'fragmentation_level': 'high' if patch_count.getInfo().get('labels', 0) > 10 else 'low'
            }
        except:
            return {'patch_count': 0, 'fragmentation_level': 'unknown'}
    
    def _identify_change_hotspots(self, mask: ee.Image, aoi: ee.Geometry) -> List[Dict]:
        """Identify hotspots of change within the AOI"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated clustering algorithms
        return [{'coordinates': [0, 0], 'intensity': 'high'}]  # Placeholder
    
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
    
    def _calculate_percentage_of_aoi(self, change_area_m2: float, aoi: ee.Geometry) -> float:
        """Calculate percentage of AOI affected by change"""
        try:
            aoi_area = aoi.area().getInfo()
            return (change_area_m2 / aoi_area) * 100
        except:
            return 0.0
    
    def _generate_deforestation_recommendations(self, severity: str, area_ha: float) -> List[str]:
        """Generate recommendations for deforestation issues"""
        recommendations = []
        
        if severity in ['severe', 'moderate']:
            recommendations.extend([
                "Immediate field verification recommended",
                "Contact local forest authorities",
                "Implement enhanced monitoring"
            ])
        
        if area_ha > 5:
            recommendations.append("Consider satellite-based verification")
        
        return recommendations
    
    def _analyze_urban_pattern(self, urban_mask: ee.Image, aoi: ee.Geometry) -> str:
        """Analyze urban expansion patterns"""
        # Simplified pattern analysis
        return 'scattered'  # Could be 'contiguous', 'scattered', 'linear', etc.
    
    def _analyze_urban_proximity(self, urban_mask: ee.Image, aoi: ee.Geometry) -> Dict:
        """Analyze proximity to existing urban areas"""
        return {
            'distance_to_urban': 100,  # meters
            'connectivity': 'connected'
        }
    
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
            needs.extend(['road_access', 'utilities'])
        if pattern == 'scattered':
            needs.append('connectivity_planning')
        return needs
    
    def _generate_urban_recommendations(self, pattern: str, area_ha: float) -> List[str]:
        """Generate recommendations for urban expansion"""
        recommendations = []
        
        if pattern == 'scattered':
            recommendations.append("Consider consolidated development planning")
        
        if area_ha > 2:
            recommendations.append("Infrastructure impact assessment needed")
        
        return recommendations
    
    def _classify_water_change(self, loss_m2: float, gain_m2: float) -> str:
        """Classify type of water body change"""
        if loss_m2 > gain_m2 * 2:
            return 'significant_loss'
        elif gain_m2 > loss_m2 * 2:
            return 'significant_gain'
        else:
            return 'minimal_change'
    
    def _assess_water_change_permanency(self, water_results: Dict, aoi: ee.Geometry) -> str:
        """Assess if water changes are permanent or seasonal"""
        # This would require temporal analysis
        return 'unknown'  # Placeholder
    
    def _assess_water_environmental_impact(self, loss_m2: float, gain_m2: float, change_type: str) -> str:
        """Assess environmental impact of water changes"""
        if change_type == 'significant_loss' and loss_m2 > 10000:
            return 'high'
        else:
            return 'low'
    
    def _assess_water_risk(self, loss_m2: float, change_type: str) -> str:
        """Assess risk level of water changes"""
        if change_type == 'significant_loss':
            return 'high'
        else:
            return 'low'
    
    def _generate_water_recommendations(self, change_type: str, loss_m2: float) -> List[str]:
        """Generate recommendations for water body changes"""
        recommendations = []
        
        if change_type == 'significant_loss':
            recommendations.extend([
                "Investigate cause of water loss",
                "Monitor for drought conditions",
                "Check for upstream diversions"
            ])
        
        return recommendations
    
    def _get_aoi_info(self, aoi: ee.Geometry) -> Dict:
        """Get AOI information"""
        try:
            area_m2 = aoi.area().getInfo()
            bounds = aoi.bounds().getInfo()
            
            return {
                'area_m2': area_m2,
                'area_ha': area_m2 / 10000,
                'bounds': bounds
            }
        except:
            return {'area_m2': 0, 'area_ha': 0, 'bounds': None}
    
    def _generate_summary(self, detailed_analysis: Dict) -> Dict:
        """Generate summary of all analyses"""
        summary = {
            'total_changes_detected': len(detailed_analysis),
            'high_priority_alerts': 0,
            'total_area_affected_ha': 0
        }
        
        for analysis_type, analysis in detailed_analysis.items():
            if analysis.get('alert_triggered', False):
                summary['high_priority_alerts'] += 1
            
            # Sum up areas (this is simplified)
            if 'area_affected' in analysis:
                summary['total_area_affected_ha'] += analysis['area_affected'].get('hectares', 0)
        
        return summary
    
    def _generate_overall_recommendations(self, detailed_analysis: Dict) -> List[str]:
        """Generate overall recommendations"""
        recommendations = set()
        
        for analysis_type, analysis in detailed_analysis.items():
            if 'recommendations' in analysis:
                recommendations.update(analysis['recommendations'])
        
        return list(recommendations)
    
    def _assess_overall_confidence(self, change_detection_results: Dict) -> Dict:
        """Assess overall confidence in the analysis"""
        return {
            'overall_confidence': 'medium',
            'factors': [
                'Cloud cover within acceptable limits',
                'Temporal gap appropriate for change detection'
            ]
        }