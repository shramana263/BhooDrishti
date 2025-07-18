from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np



def display_summary_results(analysis_report: dict):
    """Display summary of analysis results"""
    
    summary = analysis_report.get('summary', {})
    alerts = analysis_report.get('alerts', [])
    detailed = analysis_report.get('detailed_analysis', {})
    cloud_assessment = analysis_report.get('cloud_assessment', {})
    
    print(f"📊 Analysis Summary:")
    print(f"   🕒 Period: {summary.get('analysis_period', 'Unknown')}")
    print(f"   🔍 Changes detected: {summary.get('total_changes_detected', 0)}")
    print(f"   🚨 High priority alerts: {summary.get('high_priority_alerts', 0)}")
    print(f"   📐 Total area affected: {summary.get('total_area_affected_ha', 0):.2f} hectares")
    print(f"   🎯 Dominant change: {summary.get('dominant_change_type', 'none')}")
    print(f"   ⚠️  Overall impact: {summary.get('overall_impact', 'low')}")
    print()
    
    # Display cloud assessment if available
    if cloud_assessment:
        print(f"☁️  Cloud Impact Assessment:")
        print(f"   📊 Image 1 cloud coverage: {cloud_assessment['cloud_statistics']['image1_coverage']:.1f}%")
        print(f"   📊 Image 2 cloud coverage: {cloud_assessment['cloud_statistics']['image2_coverage']:.1f}%")
        print(f"   📈 Coverage difference: {cloud_assessment['cloud_statistics']['coverage_difference']:.1f}%")
        print(f"   🎯 Impact level: {cloud_assessment['impact_level'].upper()}")
        print(f"   ✅ Analysis reliable: {'YES' if cloud_assessment['analysis_reliable'] else 'NO'}")
        
        if cloud_assessment.get('warnings'):
            print(f"   ⚠️  Warnings:")
            for warning in cloud_assessment['warnings']:
                print(f"      • {warning}")
        
        if cloud_assessment.get('limitations'):
            print(f"   📋 Limitations:")
            for limitation in cloud_assessment['limitations']:
                print(f"      • {limitation}")
        print()
    
    # Display specific findings
    if 'deforestation' in detailed:
        defor = detailed['deforestation']
        print(f"🌲 Deforestation Analysis:")
        print(f"   📊 Area lost: {defor['area_affected']['hectares']:.2f} ha")
        print(f"   ⚠️  Severity: {defor['severity']}")
        print(f"   🎯 Risk level: {defor['risk_level']}")
        print()
    
    if 'urban_expansion' in detailed:
        urban = detailed['urban_expansion']
        print(f"🏙️  Urban Expansion Analysis:")
        print(f"   📊 Area expanded: {urban['area_expanded']['hectares']:.2f} ha")
        print(f"   🏗️  Pattern: {urban['expansion_pattern']}")
        print(f"   🌍 Environmental impact: {urban['environmental_impact']}")
        print()
    
    if 'water_changes' in detailed:
        water = detailed['water_changes']
        print(f"💧 Water Body Analysis:")
        print(f"   📉 Water lost: {water['water_loss']['hectares']:.2f} ha")
        print(f"   📈 Water gained: {water['water_gain']['hectares']:.2f} ha")
        print(f"   🔄 Net change: {water['net_change']['hectares']:.2f} ha")
        print()
    
    # Display alerts
    if alerts:
        print(f"🚨 Active Alerts ({len(alerts)}):")
        for i, alert in enumerate(alerts, 1):
            print(f"   {i}. {alert['type'].replace('_', ' ').title()}")
            print(f"      Priority: {alert['priority']}")
            if 'area' in alert:
                if 'hectares' in alert['area']:
                    print(f"      Area: {alert['area']['hectares']:.2f} ha")


def create_dashboard_visualization(img1_data, img2_data, change_results, analysis_report, output_dir):
    """Create a comprehensive dashboard visualization"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Original images
    ax1 = plt.subplot(3, 4, 1)
    rgb1 = np.transpose(img1_data['original_rgb'], (1, 2, 0))
    ax1.imshow(rgb1)
    ax1.set_title('2014 - Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    rgb2 = np.transpose(img2_data['original_rgb'], (1, 2, 0))
    ax2.imshow(rgb2)
    ax2.set_title('2022 - Original Image', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # NDVI images
    if 'vegetation' in change_results:
        veg = change_results['vegetation']
        
        ax3 = plt.subplot(3, 4, 3)
        im3 = ax3.imshow(veg['ndvi_before'], cmap='RdYlGn', vmin=-1, vmax=1)
        ax3.set_title('2014 - NDVI', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = plt.subplot(3, 4, 4)
        im4 = ax4.imshow(veg['ndvi_after'], cmap='RdYlGn', vmin=-1, vmax=1)
        ax4.set_title('2022 - NDVI', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        # NDVI Change
        ax5 = plt.subplot(3, 4, 5)
        im5 = ax5.imshow(veg['change_image'], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax5.set_title('NDVI Change\n(Red: Loss, Blue: Gain)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # Vegetation changes combined
        ax6 = plt.subplot(3, 4, 6)
        veg_change = np.zeros_like(veg['deforestation_mask'], dtype=np.float32)
        veg_change[veg['deforestation_mask']] = -1
        veg_change[veg['afforestation_mask']] = 1
        im6 = ax6.imshow(veg_change, cmap='RdYlGn', vmin=-1, vmax=1)
        ax6.set_title('Vegetation Changes\n(Red: Deforestation)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Urban expansion
    if 'urban' in change_results:
        urban = change_results['urban']
        
        ax7 = plt.subplot(3, 4, 7)
        ax7.imshow(urban['urban_expansion_mask'], cmap='Reds')
        ax7.set_title('Urban Expansion', fontsize=12, fontweight='bold')
        ax7.axis('off')
    
    # Water changes
    if 'water' in change_results:
        water = change_results['water']
        
        ax8 = plt.subplot(3, 4, 8)
        water_change = np.zeros_like(water['water_loss_mask'], dtype=np.float32)
        water_change[water['water_loss_mask']] = -1
        water_change[water['water_gain_mask']] = 1
        im8 = ax8.imshow(water_change, cmap='RdBu', vmin=-1, vmax=1)
        ax8.set_title('Water Changes\n(Red: Loss, Blue: Gain)', fontsize=12, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # Summary statistics text
    ax9 = plt.subplot(3, 4, (9, 12))
    ax9.axis('off')
    
    # Create summary text
    summary = analysis_report.get('summary', {})
    detailed = analysis_report.get('detailed_analysis', {})
    alerts = analysis_report.get('alerts', [])
    
    # Fix analysis period
    metadata = analysis_report.get('metadata', {})
    analysis_period = metadata.get('analysis_period', 'Unknown')
    
    summary_text = f"""
BhooDrishti Change Detection Analysis
Period: {analysis_period}

SUMMARY STATISTICS:
• Total area affected: {summary.get('total_area_affected_ha', 0):.2f} hectares
• High priority alerts: {summary.get('high_priority_alerts', 0)}
• Overall impact level: {summary.get('overall_impact', 'low').upper()}

DETAILED FINDINGS:
"""
    
    if 'deforestation' in detailed:
        defor = detailed['deforestation']
        summary_text += f"""
DEFORESTATION:
• Area lost: {defor['area_affected']['hectares']:.2f} ha
• Severity: {defor['severity'].upper()}
• Risk level: {defor['risk_level'].upper()}
• Alert: {'TRIGGERED' if defor['alert_triggered'] else 'Not triggered'}
"""
    
    if 'urban_expansion' in detailed:
        urban = detailed['urban_expansion']
        summary_text += f"""
URBAN EXPANSION:
• Area expanded: {urban['area_expanded']['hectares']:.2f} ha
• Pattern: {urban['expansion_pattern'].upper()}
• Environmental impact: {urban['environmental_impact'].upper()}
• Alert: {'TRIGGERED' if urban['alert_triggered'] else 'Not triggered'}
"""
    
    if 'water_changes' in detailed:
        water = detailed['water_changes']
        summary_text += f"""
WATER CHANGES:
• Water lost: {water['water_loss']['hectares']:.2f} ha
• Water gained: {water['water_gain']['hectares']:.2f} ha
• Net change: {water['net_change']['hectares']:.2f} ha
• Risk level: {water['risk_assessment'].upper()}
• Alert: {'TRIGGERED' if water['alert_triggered'] else 'Not triggered'}
"""
    
    if alerts:
        summary_text += f"\nACTIVE ALERTS: {len(alerts)}\n"
        for alert in alerts:
            summary_text += f"• {alert['type'].replace('_', ' ').title()} ({alert['priority']} priority)\n"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='sans-serif',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Main title
    fig.suptitle('BhooDrishti Report\nWest Bengal State', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    
    # Save dashboard
    dashboard_path = output_dir / "change_detection_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    
    # Don't show plot in non-interactive environment
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory
    
    print(f"📊 Dashboard saved: {dashboard_path}")


def create_cloud_interference_dashboard(img1_data, img2_data, cloud_impact, output_dir, image1_path, image2_path):
    """
    Create a comprehensive dashboard for cloud interference analysis termination
    
    Args:
        img1_data: First image data
        img2_data: Second image data  
        cloud_impact: Cloud impact assessment results
        output_dir: Output directory path
        image1_path: Path to first image
        image2_path: Path to second image
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Original images with cloud masks
    ax1 = plt.subplot(3, 4, 1)
    rgb1 = np.transpose(img1_data['original_rgb'], (1, 2, 0))
    ax1.imshow(rgb1)
    ax1.set_title(f'Image 1: {Path(image1_path).name}\nOriginal RGB', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    rgb2 = np.transpose(img2_data['original_rgb'], (1, 2, 0))
    ax2.imshow(rgb2)
    ax2.set_title(f'Image 2: {Path(image2_path).name}\nOriginal RGB', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Cloud masks
    ax3 = plt.subplot(3, 4, 3)
    cloud_mask1 = img1_data['cloud_info']['cloud_mask']
    ax3.imshow(cloud_mask1, cmap='Blues', alpha=0.8)
    ax3.set_title(f'Image 1: Cloud Mask\nCoverage: {cloud_impact["cloud_statistics"]["image1_coverage"]:.1f}%', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 4)
    cloud_mask2 = img2_data['cloud_info']['cloud_mask']
    ax4.imshow(cloud_mask2, cmap='Blues', alpha=0.8)
    ax4.set_title(f'Image 2: Cloud Mask\nCoverage: {cloud_impact["cloud_statistics"]["image2_coverage"]:.1f}%', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Combined cloud coverage visualization
    ax5 = plt.subplot(3, 4, 5)
    combined_clouds = np.maximum(cloud_mask1, cloud_mask2)
    ax5.imshow(combined_clouds, cmap='Reds', alpha=0.8)
    ax5.set_title('Combined Cloud Coverage\n(Areas affected in either image)', 
                  fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Cloud difference visualization
    ax6 = plt.subplot(3, 4, 6)
    cloud_diff = cloud_mask2.astype(float) - cloud_mask1.astype(float)
    im6 = ax6.imshow(cloud_diff, cmap='RdBu', vmin=-1, vmax=1)
    ax6.set_title('Cloud Difference\n(Red: More clouds in Image 2)', 
                  fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Reliability assessment visualization
    ax7 = plt.subplot(3, 4, 7)
    # Create a reliability map based on cloud coverage
    reliability_map = np.ones_like(cloud_mask1, dtype=float)
    reliability_map[combined_clouds > 0] = 0.3  # Low reliability in cloudy areas
    im7 = ax7.imshow(reliability_map, cmap='RdYlGn', vmin=0, vmax=1)
    ax7.set_title('Analysis Reliability Map\n(Green: Reliable, Red: Unreliable)', 
                  fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    # Cloud statistics pie chart
    ax8 = plt.subplot(3, 4, 8)
    cloud_stats = cloud_impact['cloud_statistics']
    img1_coverage = cloud_stats['image1_coverage']
    img2_coverage = cloud_stats['image2_coverage']
    
    # Create pie chart data
    labels = ['Image 1 Clear', 'Image 1 Cloudy', 'Image 2 Clear', 'Image 2 Cloudy']
    sizes = [100-img1_coverage, img1_coverage, 100-img2_coverage, img2_coverage]
    colors = ['lightgreen', 'lightcoral', 'lightblue', 'darkred']
    
    wedges, texts, autotexts = ax8.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax8.set_title('Cloud Coverage Comparison', fontsize=12, fontweight='bold')
    
    # Make text smaller
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    # Detailed analysis text
    ax9 = plt.subplot(3, 4, (9, 12))
    ax9.axis('off')
    
    # Create analysis termination summary text
    termination_text = f"""
🚫 ANALYSIS TERMINATED DUE TO CLOUD INTERFERENCE

CLOUD IMPACT ASSESSMENT:
• Impact Level: {cloud_impact['impact_assessment'].upper()}
• Analysis Reliable: {'NO' if not cloud_impact['analysis_reliable'] else 'YES'}
• Coverage Difference: {cloud_impact['cloud_statistics']['coverage_difference']:.1f}%

CLOUD STATISTICS:
• Image 1 Cloud Coverage: {cloud_impact['cloud_statistics']['image1_coverage']:.1f}%
• Image 2 Cloud Coverage: {cloud_impact['cloud_statistics']['image2_coverage']:.1f}%
• Maximum Coverage: {cloud_impact['cloud_statistics']['max_coverage']:.1f}%
• Minimum Coverage: {cloud_impact['cloud_statistics']['min_coverage']:.1f}%

ANALYSIS LIMITATIONS:
"""
    
    for limitation in cloud_impact.get('limitations', []):
        termination_text += f"• {limitation}\n"
    
    termination_text += "\nRECOMMENDations:\n"
    for recommendation in cloud_impact.get('analysis_recommendations', []):
        termination_text += f"• {recommendation}\n"
    
    if cloud_impact.get('warning_messages'):
        termination_text += "\n⚠️ WARNINGS:\n"
        for warning in cloud_impact['warning_messages']:
            termination_text += f"• {warning}\n"
    
    termination_text += f"""
CONCLUSION:
Change detection analysis cannot be performed reliably due to significant 
cloud interference. The difference in cloud coverage between images 
({cloud_impact['cloud_statistics']['coverage_difference']:.1f}%) would lead to 
false change detections and unreliable results.

NEXT STEPS:
• Acquire images with better atmospheric conditions
• Use cloud-free image pairs from the same season
• Consider using radar data for all-weather monitoring
• Apply advanced cloud removal algorithms if available
"""
    
    ax9.text(0.05, 0.95, termination_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose", alpha=0.9))
    
    # Main title
    fig.suptitle('BhooDrishti Cloud Interference Report\nAnalysis Terminated Due to Poor Image Quality', 
                 fontsize=16, fontweight='bold', y=0.98, color='darkred')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    
    # Save dashboard
    dashboard_path = Path(output_dir) / "cloud_interference_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    
    # Don't show plot in non-interactive environment
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    
    plt.close()  # Close figure to free memory
    
    print(f"📊 Cloud interference dashboard saved: {dashboard_path}")
    
    # Also save a summary report
    summary_report_path = Path(output_dir) / "cloud_interference_report.txt"
    with open(summary_report_path, 'w') as f:
        f.write("BhooDrishti Cloud Interference Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {cloud_impact.get('analysis_date', 'Unknown')}\n")
        f.write(f"Image 1: {Path(image1_path).name}\n")
        f.write(f"Image 2: {Path(image2_path).name}\n\n")
        f.write("TERMINATION REASON: Cloud interference makes change detection unreliable\n\n")
        f.write(termination_text.replace('🚫', '').replace('⚠️', 'WARNING:'))
    
    print(f"📄 Text report saved: {summary_report_path}")

