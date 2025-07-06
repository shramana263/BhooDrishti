import ee
import logging
import yaml
import json
from datetime import datetime, timedelta
from datetime import datetime
from typing import Dict, List
from src.change_detection import ChangeDetectionEngine
from src.change_analysis import ChangeAnalysisEngine
from src.data_retrieval import GEEDataRetriever
from src.preprocessing import SatellitePreprocessor

class ChangeDetectionPipeline:
    """
    Main pipeline for change detection and analysis
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize components with proper config handling
        self.data_retrieval = GEEDataRetriever(
            service_account_path=self.config.get('gee', {}).get('service_account_path'),
            config_path=config_path
        )
        self.preprocessor = SatellitePreprocessor(config_path)
        self.change_detector = ChangeDetectionEngine(self.config)
        self.change_analyzer = ChangeAnalysisEngine(self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def run_change_detection(self, aoi_geojson: Dict, start_date: str, 
                       end_date: str, change_types: List[str] = None) -> Dict:
        """
        Run complete change detection pipeline with enhanced error handling
        """
        try:
            # Convert AOI to EE Geometry
            aoi = ee.Geometry(aoi_geojson)
            
            # Validate AOI size
            aoi_area = aoi.area().getInfo()
            print(f"📍 AOI area: {aoi_area/1000000:.2f} km²")
            
            if aoi_area > 1000000000:  # 1000 km²
                print("⚠️ Warning: Large AOI may cause performance issues")
            
            # Split date range into before/after periods
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            mid_dt = start_dt + (end_dt - start_dt) / 2
            
            print("🔍 Getting satellite collections...")
            # Get image collections
            before_collection = self.data_retrieval.get_satellite_collection(
                'sentinel2', start_date, mid_dt.isoformat(), aoi
            )
            after_collection = self.data_retrieval.get_satellite_collection(
                'sentinel2', mid_dt.isoformat(), end_date, aoi
            )
            
            print("📸 Creating composite images...")
            # Create composite images
            before_image = self.data_retrieval.process_image_collection(
                before_collection, 'sentinel2', aoi
            )
            after_image = self.data_retrieval.process_image_collection(
                after_collection, 'sentinel2', aoi
            )
            
            print("📊 Calculating spectral indices...")
            # Add spectral indices
            before_processed = self.data_retrieval.calculate_indices(before_image)
            after_processed = self.data_retrieval.calculate_indices(after_image)
            
            # Debug: Check available bands before processing
            print(f"🔍 Available bands before processing: {before_processed.bandNames().getInfo()}")
            
            # Select all available bands needed for change detection and cast to float
            # Include NIR and SWIR bands for vegetation and other indices
            try:
                # Try to select all standard bands first
                before_processed = before_processed.select(['red', 'green', 'blue', 'nir', 'swir1', 'swir2']).multiply(0.0001)
                after_processed = after_processed.select(['red', 'green', 'blue', 'nir', 'swir1', 'swir2']).multiply(0.0001)
                print("✅ Successfully selected all bands including NIR and SWIR")
            except Exception as band_error:
                print(f"⚠️ Could not select all bands: {band_error}")
                # Fall back to available bands
                available_bands = before_processed.bandNames().getInfo()
                common_bands = ['red', 'green', 'blue']
                if 'nir' in available_bands:
                    common_bands.append('nir')
                if 'swir1' in available_bands:
                    common_bands.append('swir1')
                if 'swir2' in available_bands:
                    common_bands.append('swir2')
                
                before_processed = before_processed.select(common_bands).multiply(0.0001)
                after_processed = after_processed.select(common_bands).multiply(0.0001)
                print(f"✅ Selected available bands: {common_bands}")

            
            print("🔬 Performing change detection...")
            # Perform change detection with debug info
            change_results = self.change_detector.comprehensive_change_detection(
                before_processed, after_processed, aoi, change_types
            )
            
            # Debug: Print some statistics
            if 'vegetation' in change_results:
                veg_stats = change_results['vegetation']['statistics']
                # print(f"📊 Deforestation area: {veg_stats['deforestation_area']}")
                # print(f"📊 Afforestation area: {veg_stats['afforestation_area']}")
            
            print("📋 Generating analysis report...")
            # Analyze changes
            metadata = {
                'start_date': start_date,
                'end_date': end_date,
                'mid_date': mid_dt.isoformat(),
                'aoi': aoi_geojson,
                'satellite': 'sentinel2'
            }
            
            analysis_report = self.change_analyzer.generate_comprehensive_report(
                change_results, aoi, metadata
            )
            
            return {
                'detection_results': change_results,
                'analysis_report': analysis_report,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in change detection pipeline: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not found"""
        return {
            'gee': {
                'service_account_path': None,
                'project_id': None,
                'max_pixels': 1000000000,
                'default_scale': 10
            },
            'change_detection': {
                'ndvi_threshold': 0.2,
                'confidence_threshold': 0.7,
                'urban_threshold': 0.1,
                'water_threshold': 0.0
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
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/change_detection.log'),
                logging.StreamHandler()
            ]
        )

def create_ee_export_task(image, aoi_geometry, prefix, error_msg):
    """Create Earth Engine export task for large images"""
    try:
        # Generate unique task name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"{prefix}_{timestamp}_{ee.String(ee.Number.random()).getInfo():.0f}"
        
        # Create export task to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=task_name,
            folder='EarthEngine_Exports',
            fileNamePrefix=task_name,
            region=aoi_geometry,
            scale=30,  # Sentinel-2 resolution
            crs='EPSG:4326',
            maxPixels=1e9
        )
        
        # Start the task
        task.start()
        
        return {
            'type': 'EE_Export_Task',
            'task_id': task.id,
            'task_name': task_name,
            'status': 'SUBMITTED',
            'export_method': 'Google Drive',
            'folder': 'EarthEngine_Exports',
            'original_error': error_msg,
            'note': 'Large image exported to Google Drive. Check Earth Engine Tasks tab.',
            'instructions': 'Go to https://code.earthengine.google.com/tasks to monitor progress'
        }
        
    except Exception as export_error:
        return {
            'type': 'EE_Export_Failed',
            'error': str(export_error),
            'original_error': error_msg,
            'note': 'Failed to create export task'
        }

def export_to_cloud_storage(image, aoi_geometry, bucket_name, prefix):
    """Export large images to Google Cloud Storage"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"{prefix}_{timestamp}"
        
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=task_name,
            bucket=bucket_name,
            fileNamePrefix=f"change_detection/{task_name}",
            region=aoi_geometry,
            scale=30,
            crs='EPSG:4326',
            maxPixels=1e9,
            fileFormat='GeoTIFF'
        )
        
        task.start()
        
        return {
            'type': 'EE_Cloud_Export',
            'task_id': task.id,
            'task_name': task_name,
            'bucket': bucket_name,
            'status': 'SUBMITTED'
        }
    except Exception as e:
        return {'error': str(e)}

def export_to_asset(image, aoi_geometry, asset_id, prefix):
    """Export images as Earth Engine assets"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"{prefix}_{timestamp}"
        full_asset_id = f"{asset_id}/{task_name}"
        
        task = ee.batch.Export.image.toAsset(
            image=image,
            description=task_name,
            assetId=full_asset_id,
            region=aoi_geometry,
            scale=30,
            crs='EPSG:4326',
            maxPixels=1e9
        )
        
        task.start()
        
        return {
            'type': 'EE_Asset_Export',
            'task_id': task.id,
            'task_name': task_name,
            'asset_id': full_asset_id,
            'status': 'SUBMITTED'
        }
    except Exception as e:
        return {'error': str(e)}

def convert_results_with_exports(obj, aoi_geometry, prefix="change_detection"):
    """Convert EE objects to exportable references with optimized parameters"""
    if hasattr(obj, 'getInfo'):  # EE object
        try:
            if hasattr(obj, 'getDownloadURL'):  # EE Image
                # Use the passed aoi_geometry (EE Geometry object) instead of aoi dict
                download_url = obj.getDownloadURL({
                    'scale': 100,
                    'crs': 'EPSG:4326',
                    'format': 'GeoTIFF',
                    'region': aoi_geometry.bounds().getInfo()  # Use the EE Geometry object
                })
                return {
                    'type': 'EE_Image',
                    'download_url': download_url,
                    'metadata': {
                        'bands': obj.bandNames().getInfo(),
                        'scale': 100,
                        'note': 'Reduced resolution for export optimization'
                    }
                }
            else:
                return obj.getInfo()
        except Exception as e:
            # For large images, create Earth Engine export task
            return create_ee_export_task(obj, aoi_geometry, prefix, str(e))
    elif isinstance(obj, dict):
        return {k: convert_results_with_exports(v, aoi_geometry, prefix) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_results_with_exports(item, aoi_geometry, prefix) for item in obj]
    else:
        return obj

def monitor_export_tasks(task_ids, check_interval=30):
    """Monitor the progress of export tasks"""
    import time
    
    print("📋 Monitoring export tasks...")
    
    while task_ids:
        completed_tasks = []
        
        for task_id in task_ids:
            try:
                task = ee.batch.Task(task_id)
                status = task.status()
                
                print(f"Task {task_id}: {status['state']}")
                
                if status['state'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    completed_tasks.append(task_id)
                    
                    if status['state'] == 'COMPLETED':
                        print(f"✅ Task {task_id} completed successfully")
                    else:
                        print(f"❌ Task {task_id} failed: {status.get('error_message', 'Unknown error')}")
                        
            except Exception as e:
                print(f"Error checking task {task_id}: {e}")
                completed_tasks.append(task_id)
        
        # Remove completed tasks
        for task_id in completed_tasks:
            task_ids.remove(task_id)
        
        if task_ids:
            print(f"⏳ {len(task_ids)} tasks still running. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)
        else:
            print("🎉 All export tasks completed!")

def get_task_list():
    """Get list of all tasks for the authenticated user"""
    tasks = ee.batch.Task.list()
    
    print("📋 Recent Export Tasks:")
    for i, task in enumerate(tasks[:10]):  # Show last 10 tasks
        status = task.status()
        print(f"{i+1}. {status['description']}: {status['state']}")
        if 'start_timestamp_ms' in status:
            start_time = datetime.fromtimestamp(status['start_timestamp_ms'] / 1000)
            print(f"   Started: {start_time}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os

    os.makedirs('logs', exist_ok=True)
    
    # Example usage
    print("🌍 Initializing Change Detection Pipeline...")
    pipeline = ChangeDetectionPipeline('config/config.yaml')
    
    aoi_dict = {
        "type": "Polygon",
        "coordinates": [
          [
            [
              88.33427370587458,
              22.584707008791398
            ],
            [
              88.33234535609432,
              22.583838484125295
            ],
            [
              88.33221993497057,
              22.5810881198899
            ],
            [
              88.3379422737479,
              22.581319731626934
            ],
            [
              88.33662535194753,
              22.584576730440276
            ],
            [
              88.33427370587458,
              22.584707008791398
            ]
          ]
        ]
    }

    print("🚀 Starting change detection analysis...")
    results = pipeline.run_change_detection(
        aoi_geojson=aoi_dict,
        start_date="2013-01-01",
        end_date="2024-12-31",
        change_types=['vegetation', 'urban', 'water']
    )
    
    print(f"✅ Change detection completed: {results['status']}")
    if results['status'] == 'success':
        alerts = results['analysis_report']['alerts']
        print(f"🚨 Alerts triggered: {len(alerts)}")
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"data/results/change_detection_{timestamp}.json"

        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Convert AOI dict to EE Geometry for exports
        aoi_geometry = ee.Geometry(aoi_dict)

        print("📊 Converting results with export URLs...")
        exportable_results = convert_results_with_exports(results, aoi_geometry)
        
        # Add processing metadata
        exportable_results['metadata'] = {
            'processing_timestamp': timestamp,
            'processing_date': datetime.now().isoformat(),
            'export_note': 'EE Images converted to download URLs or export tasks'
        }

        # Save the results to a JSON file
        with open(output_filename, 'w') as f:
            json.dump(exportable_results, f, indent=4)

        print(f"📄 Results saved to {output_filename}")
        
        # Print export tasks created
        print("\n📤 Export Tasks Created:")
        for key, value in exportable_results.get('detection_results', {}).items():
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and sub_value.get('type') == 'EE_Export_Task':
                    print(f"   - {key}.{sub_key}: Task {sub_value['task_name']}")
                    
        for alert in alerts:
            print(f"   - {alert['type']}: {alert.get('severity', 'N/A')}")
    else:
        print(f"❌ Error: {results['error']}")