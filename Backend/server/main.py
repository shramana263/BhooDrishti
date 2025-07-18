from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from typing import Union, List, Optional
import os
import sys
from pathlib import Path
import tempfile
import json

# Add the PNG adaptation modules to path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
png_adaptation_dir = backend_dir / "png_adaptation"
sys.path.insert(0, str(png_adaptation_dir))

# Import modules directly
try:
    from png_processor import PNGSatelliteProcessor
    from png_change_detection import PNGChangeDetectionEngine
    from png_change_analysis import PNGChangeAnalysisEngine
    from dashboard_maker.dashboard_maker import create_cloud_interference_dashboard, create_dashboard_visualization, display_summary_results
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

app = FastAPI(title="BhooDrishti Backend", description="Satellite Image Change Detection API")

# Create output directory
OUTPUT_DIR = Path("/tmp/bhooDrishti_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def read_root():
    return {"message": "Hello From BhooDrishti Backend"} 

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "BhooDrishti Backend"}

@app.post("/predict")
async def analyze_image(
    image1: UploadFile = File(...), 
    image2: UploadFile = File(...),
    config: Optional[str] = Form(None),
    change_types: Optional[str] = Form(None)
):
    """
    Analyze two satellite images for change detection
    
    Args:
        image1: First satellite image (earlier time)
        image2: Second satellite image (later time)  
        config: Optional configuration as JSON string
        change_types: Optional comma-separated change types
        
    Returns:
        Analysis results with file paths to generated reports and visualizations
    """
    
    temp_files = []
    print("Config: ",config)
    print("Change Types: ", change_types)
    try:
        # Validate inputs
        if not image1.filename or not image2.filename:
            return {"error": "Both images are required"}
        
        # Parse configuration
        if config is None:
            print("Using default configuration")
            config_dict = {
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
        else:
            try:
                config_dict = json.loads(config)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid config format: {str(e)}"}
        
        # Parse change types
        if change_types is None:
            print("Using default change types: vegetation, urban, water")
            change_types_list = ['vegetation', 'urban', 'water']
        else:
            change_types_list = [ct.strip() for ct in change_types.split(',')]
        
        # Create temporary files for images
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1:
            content1 = await image1.read()
            tmp1.write(content1)
            image1_path = tmp1.name
            temp_files.append(image1_path)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
            content2 = await image2.read()
            tmp2.write(content2)
            image2_path = tmp2.name
            temp_files.append(image2_path)
        
        # Create unique output directory for this analysis
        import uuid
        analysis_id = str(uuid.uuid4())[:8]
        analysis_output_dir = OUTPUT_DIR / f"analysis_{analysis_id}"
        analysis_output_dir.mkdir(exist_ok=True)
        
        # Initialize processing engines
        processor = PNGSatelliteProcessor(config_dict)
        change_detection = PNGChangeDetectionEngine(config_dict)  # Fixed: pass config directly
        change_analysis = PNGChangeAnalysisEngine(config_dict)    # Fixed: pass config directly
        
        # Load and process images
        print(f"Loading images: {image1.filename}, {image2.filename}")
        img1_data = processor.load_png_image(image1_path)
        img2_data = processor.load_png_image(image2_path)
        
        if img1_data is None or img2_data is None:
            return {"error": "Failed to process one or both images. Please check image format and content."}
        
        # Assess cloud impact
        cloud_impact = processor.assess_cloud_impact_for_analysis(
            img1_data['cloud_info'], 
            img2_data['cloud_info']
        )
        
        # Check if analysis should proceed due to cloud interference
        if not cloud_impact['analysis_reliable']:
            print("Analysis terminated due to cloud interference")
            
            # Create cloud interference dashboard
            create_cloud_interference_dashboard(
                img1_data=img1_data, 
                img2_data=img2_data, 
                cloud_impact=cloud_impact,
                output_dir=analysis_output_dir,
                image1_path=image1_path, 
                image2_path=image2_path
            )
            
            return {
                "status": "analysis_terminated",
                "reason": "cloud_interference", 
                "message": "Cloud interference detected. Analysis cannot be performed reliably.",
                "cloud_statistics": cloud_impact['cloud_statistics'],
                "impact_level": cloud_impact['impact_assessment'],
                "analysis_id": analysis_id,
                "files": {
                    "cloud_dashboard": f"/download/{analysis_id}/cloud_interference_dashboard.png",
                    "cloud_report": f"/download/{analysis_id}/cloud_interference_report.txt"
                },
                "recommendations": cloud_impact['analysis_recommendations']
            }
        
        # Proceed with change detection analysis
        print("Proceeding with change detection analysis")
        
        # Perform comprehensive change detection
        change_results = change_detection.comprehensive_change_detection_png(
            image1_path, 
            image2_path, 
            change_types=change_types_list,
            cloud_impact_info=cloud_impact
        )
        
        # Generate detailed analysis report
        analysis_report = change_analysis.generate_comprehensive_report_png(
            change_results, 
            image1_path, 
            image2_path
        )
        
        # Create visualizations
        change_detection.visualize_change_results(
            change_results, 
            image1_path, 
            image2_path, 
            str(analysis_output_dir)
        )
        
        # Create RGB comparison
        processor.visualize_png_analysis(
            image1_path, 
            image2_path, 
            str(analysis_output_dir / "rgb_comparison.png")
        )
        
        # Export comprehensive report
        report_path = change_analysis.export_analysis_report(
            analysis_report, 
            str(analysis_output_dir / "comprehensive_analysis_report.json")
        )
        
        # Create dashboard visualization
        create_dashboard_visualization(
            img1_data=img1_data, 
            img2_data=img2_data, 
            change_results=change_results, 
            analysis_report=analysis_report, 
            output_dir=analysis_output_dir
        )
        
        # Prepare response with file paths
        summary = analysis_report.get('summary', {})
        alerts = analysis_report.get('alerts', [])
        
        # Build response
        response = {
            "status": "success",
            "analysis_id": analysis_id,
            "message": "Change detection analysis completed successfully",
            "summary": {
                "analysis_period": summary.get('analysis_period', 'Unknown'),
                "total_changes_detected": summary.get('total_changes_detected', 0),
                "high_priority_alerts": summary.get('high_priority_alerts', 0),
                "total_area_affected_ha": round(summary.get('total_area_affected_ha', 0), 2),
                "dominant_change_type": summary.get('dominant_change_type', 'none'),
                "overall_impact": summary.get('overall_impact', 'low')
            },
            "cloud_assessment": {
                "impact_level": cloud_impact['impact_assessment'],
                "analysis_reliable": cloud_impact['analysis_reliable'],
                "coverage_difference": cloud_impact['cloud_statistics']['coverage_difference']
            } if cloud_impact['impact_assessment'] != 'minimal' else None,
            "alerts": [
                {
                    "type": alert['type'],
                    "priority": alert['priority'],
                    "area_ha": alert.get('area', {}).get('hectares', 0)
                } for alert in alerts
            ],
            "files": {
                "dashboard": f"/download/{analysis_id}/change_detection_dashboard.png",
                "comprehensive_report": f"/download/{analysis_id}/comprehensive_analysis_report.json",
                "rgb_comparison": f"/download/{analysis_id}/rgb_comparison.png",
                "ndvi_change": f"/download/{analysis_id}/ndvi_change.png",
                "urban_expansion": f"/download/{analysis_id}/urban_expansion.png",
                "comprehensive_visualization": f"/download/{analysis_id}/comprehensive_change_detection.png"
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred during analysis: {str(e)}"}
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not clean up temp file {temp_file}: {e}")

@app.get("/download/{analysis_id}/{filename}")
async def download_file(analysis_id: str, filename: str):
    """
    Download generated analysis files
    
    Args:
        analysis_id: Unique analysis identifier
        filename: Name of the file to download
        
    Returns:
        File response with the requested file
    """
    try:
        file_path = OUTPUT_DIR / f"analysis_{analysis_id}" / filename
        
        if not file_path.exists():
            return {"error": f"File not found: {filename}"}
        
        # Determine media type based on file extension
        if filename.endswith('.png'):
            media_type = 'image/png'
        elif filename.endswith('.json'):
            media_type = 'application/json'
        elif filename.endswith('.txt'):
            media_type = 'text/plain'
        else:
            media_type = 'application/octet-stream'
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type
        )
        
    except Exception as e:
        return {"error": f"Error downloading file: {str(e)}"}

@app.get("/analyses")
async def list_analyses():
    """
    List all available analyses
    
    Returns:
        List of analysis IDs and their creation times
    """
    try:
        analyses = []
        
        for analysis_dir in OUTPUT_DIR.glob("analysis_*"):
            if analysis_dir.is_dir():
                analysis_id = analysis_dir.name.replace("analysis_", "")
                creation_time = analysis_dir.stat().st_mtime
                
                # Check for key files
                has_dashboard = (analysis_dir / "change_detection_dashboard.png").exists()
                has_report = (analysis_dir / "comprehensive_analysis_report.json").exists()
                has_cloud_interference = (analysis_dir / "cloud_interference_dashboard.png").exists()
                
                analyses.append({
                    "analysis_id": analysis_id,
                    "creation_time": creation_time,
                    "status": "cloud_interference" if has_cloud_interference else "completed" if has_dashboard else "incomplete",
                    "has_dashboard": has_dashboard,
                    "has_report": has_report,
                    "has_cloud_interference": has_cloud_interference
                })
        
        # Sort by creation time (newest first)
        analyses.sort(key=lambda x: x['creation_time'], reverse=True)
        
        return {
            "analyses": analyses,
            "total_count": len(analyses)
        }
        
    except Exception as e:
        return {"error": f"Error listing analyses: {str(e)}"}

@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete an analysis and all its files
    
    Args:
        analysis_id: Unique analysis identifier
        
    Returns:
        Deletion status
    """
    try:
        analysis_dir = OUTPUT_DIR / f"analysis_{analysis_id}"
        
        if not analysis_dir.exists():
            return {"error": f"Analysis not found: {analysis_id}"}
        
        # Remove all files in the analysis directory
        import shutil
        shutil.rmtree(analysis_dir)
        
        return {
            "message": f"Analysis {analysis_id} deleted successfully",
            "analysis_id": analysis_id
        }
        
    except Exception as e:
        return {"error": f"Error deleting analysis: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
