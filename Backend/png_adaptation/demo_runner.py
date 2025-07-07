"""
PNG Demo Runner
Complete demonstration of PNG-based change detection using existing system
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add the current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

# Import PNG adaptation modules
from png_processor import PNGSatelliteProcessor
from png_change_detection import PNGChangeDetectionEngine
from png_change_analysis import PNGChangeAnalysisEngine


def run_complete_png_demo():
    """
    Run complete PNG change detection demo
    Demonstrates the full workflow from PNG processing to analysis
    """
    print("🚀 BhooDrishti PNG Change Detection Demo")
    print("=" * 60)
    
    # Configuration
    config = {
        'change_detection': {
            'ndvi_threshold': 0.1,
            'urban_threshold': 0.05,
            'water_threshold': 0.3,
            'confidence_threshold': 0.5
        },
        'analysis': {
            'significant_change_area': 1000,
            'alert_thresholds': {
                'deforestation': 5000,
                'urban_expansion': 2000,
                'water_loss': 3000
            }
        }
    }
    
    # Initialize engines
    processor = PNGSatelliteProcessor(config)
    change_detector = PNGChangeDetectionEngine(config)
    analyzer = PNGChangeAnalysisEngine(config)
    
    # Define image paths
    data_dir = Path("/home/parambrata-ghosh/Development/Personal/Hackathon/ISRO/BhooDristi/Backend/data/raw")
    image1_path = data_dir / "kpc_2014.png"
    image2_path = data_dir / "kpc_2022.png"
    
    # Output directory
    output_dir = Path("/home/parambrata-ghosh/Development/Personal/Hackathon/ISRO/BhooDristi/Backend/png_adaptation/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify images exist
    if not image1_path.exists() or not image2_path.exists():
        print(f"❌ Error: Images not found!")
        print(f"   Expected: {image1_path}")
        print(f"   Expected: {image2_path}")
        return
    
    print(f"📂 Input Images:")
    print(f"   2014: {image1_path.name}")
    print(f"   2022: {image2_path.name}")
    print(f"📁 Output Directory: {output_dir}")
    print()
    
    try:
        # Step 1: Process PNG images
        print("🔄 Step 1: Processing PNG Images")
        print("-" * 40)
        
        img1_data = processor.load_png_image(str(image1_path))
        img2_data = processor.load_png_image(str(image2_path))
        
        if not img1_data or not img2_data:
            print("❌ Failed to load PNG images")
            return
        
        print("✅ PNG images loaded and processed successfully")
        print()
        
        # Step 2: Perform change detection
        print("🔍 Step 2: Change Detection Analysis")
        print("-" * 40)
        
        change_results = change_detector.comprehensive_change_detection_png(
            str(image1_path), str(image2_path),
            change_types=['vegetation', 'urban', 'water']
        )
        
        print("✅ Change detection completed")
        print()
        
        # Step 3: Detailed analysis
        print("📊 Step 3: Detailed Change Analysis")
        print("-" * 40)
        
        analysis_report = analyzer.generate_comprehensive_report_png(
            change_results, str(image1_path), str(image2_path)
        )
        
        print("✅ Detailed analysis completed")
        print()
        
        # Step 4: Generate visualizations
        print("📈 Step 4: Creating Visualizations")
        print("-" * 40)
        
        change_detector.visualize_change_results(
            change_results, str(image1_path), str(image2_path), str(output_dir)
        )
        
        # Create RGB comparison
        processor.visualize_png_analysis(
            str(image1_path), str(image2_path), 
            str(output_dir / "rgb_comparison.png")
        )
        
        print("✅ Visualizations created")
        print()
        
        # Step 5: Export analysis report
        print("📄 Step 5: Exporting Analysis Report")
        print("-" * 40)
        
        report_path = analyzer.export_analysis_report(
            analysis_report, str(output_dir / "comprehensive_analysis_report.json")
        )
        
        print("✅ Analysis report exported")
        print()
        
        # Step 6: Display summary results
        print("📋 Step 6: Summary Results")
        print("-" * 40)
        
        display_summary_results(analysis_report)
        
        # Step 7: Create final dashboard visualization
        print("🎨 Step 7: Creating Dashboard Visualization")
        print("-" * 40)
        
        create_dashboard_visualization(
            img1_data, img2_data, change_results, analysis_report, output_dir
        )
        
        print("✅ Dashboard visualization created")
        print()
        
        print("🎉 PNG Change Detection Demo Completed Successfully!")
        print("=" * 60)
        print(f"📁 All outputs saved to: {output_dir}")
        print(f"📄 Analysis report: {report_path}")
        print(f"📊 Visualizations: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()


def display_summary_results(analysis_report: dict):
    """Display summary of analysis results"""
    
    summary = analysis_report.get('summary', {})
    alerts = analysis_report.get('alerts', [])
    detailed = analysis_report.get('detailed_analysis', {})
    
    print(f"📊 Analysis Summary:")
    print(f"   🕒 Period: {summary.get('analysis_period', 'Unknown')}")
    print(f"   🔍 Changes detected: {summary.get('total_changes_detected', 0)}")
    print(f"   🚨 High priority alerts: {summary.get('high_priority_alerts', 0)}")
    print(f"   📐 Total area affected: {summary.get('total_area_affected_ha', 0):.2f} hectares")
    print(f"   🎯 Dominant change: {summary.get('dominant_change_type', 'none')}")
    print(f"   ⚠️  Overall impact: {summary.get('overall_impact', 'low')}")
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
    fig.suptitle('BhooDrishti PNG Change Detection Dashboard\nKarnataka State (2014-2022)', 
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


if __name__ == "__main__":
    run_complete_png_demo()
