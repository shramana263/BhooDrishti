#!/usr/bin/env python3
"""
Debug Script for Satellite Data Retrieval
===========================================

This script helps you:
1. Test Google Earth Engine authentication
2. Get satellite images for specific locations
3. Download images as GeoTIFF files for analysis
4. Debug the data retrieval process step by step

Usage:
    python debug_satellite_data.py
"""

import ee
import geemap
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class SatelliteDataDebugger:
    """Simple debugger for satellite data retrieval"""
    
    def __init__(self):
        self.initialize_gee()
        
    def initialize_gee(self):
        """Initialize Google Earth Engine"""
        try:
            # Try service account authentication first
            service_account = 'credentials/gee-service-account.json'
            if os.path.exists(service_account):
                credentials = ee.ServiceAccountCredentials(
                    'gee-service-account@composite-depot-428316-f1.iam.gserviceaccount.com',
                    service_account
                )
                ee.Initialize(credentials)
                print("✅ GEE initialized with service account")
            else:
                # Fallback to OAuth
                ee.Initialize()
                print("✅ GEE initialized with OAuth")
                
        except Exception as e:
            print(f"❌ GEE initialization failed: {e}")
            print("🔧 Try running: earthengine authenticate")
            raise
    
    def create_test_aoi(self, location_name="kolkata"):
        """Create test AOI for different locations"""
        locations = {
            "kolkata": {
                "coords": [[88.3476, 22.5726], [88.3676, 22.5726], 
                          [88.3676, 22.5926], [88.3476, 22.5926], [88.3476, 22.5726]],
                "description": "Small area in Kolkata, India"
            },
            "bangalore": {
                "coords": [[77.5946, 12.9716], [77.6146, 12.9716],
                          [77.6146, 12.9916], [77.5946, 12.9916], [77.5946, 12.9716]],
                "description": "Small area in Bangalore, India"
            },
            "delhi": {
                "coords": [[77.2090, 28.6139], [77.2290, 28.6139],
                          [77.2290, 28.6339], [77.2090, 28.6339], [77.2090, 28.6139]],
                "description": "Small area in Delhi, India"
            },
            "custom": {
                "coords": None,  # Will be set by user
                "description": "Custom location"
            }
        }
        
        if location_name not in locations:
            print(f"Available locations: {list(locations.keys())}")
            location_name = "kolkata"
        
        location = locations[location_name]
        if location['coords']:
            geometry = ee.Geometry.Polygon([location['coords']])
            print(f"📍 Using {location_name}: {location['description']}")
            return geometry
        else:
            print("❌ Custom location coordinates not provided")
            return None
    
    def get_satellite_image(self, aoi, start_date, end_date, satellite='sentinel2'):
        """Get satellite image for specific time period"""
        try:
            print(f"\n🛰️ Getting {satellite} imagery from {start_date} to {end_date}")
            
            # Collection IDs
            collections = {
                'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
                'landsat8': 'LANDSAT/LC08/C02/T1_L2',
                'landsat9': 'LANDSAT/LC09/C02/T1_L2'
            }
            
            if satellite not in collections:
                raise ValueError(f"Unsupported satellite: {satellite}")
            
            # Get image collection
            collection = ee.ImageCollection(collections[satellite]) \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            # Check if any images found
            count = collection.size().getInfo()
            print(f"📊 Found {count} images in collection")
            
            if count == 0:
                print("❌ No images found for the specified criteria")
                return None
            
            # Get image info
            image_list = collection.limit(5).getInfo()['features']
            print("\n📋 Available images:")
            for i, img in enumerate(image_list):
                props = img['properties']
                print(f"  {i+1}. {props.get('PRODUCT_ID', 'Unknown')} - "
                      f"Cloud: {props.get('CLOUDY_PIXEL_PERCENTAGE', 'N/A'):.1f}%")
            
            # Create median composite
            median_image = collection.median()
            
            # Add band mapping for Sentinel-2
            if satellite == 'sentinel2':
                band_map = {
                    'red': 'B4', 'green': 'B3', 'blue': 'B2', 'nir': 'B8',
                    'swir1': 'B11', 'swir2': 'B12'
                }
                # Rename bands
                median_image = median_image.select(
                    list(band_map.values()), 
                    list(band_map.keys())
                )
            
            print(f"✅ Created median composite with bands: {median_image.bandNames().getInfo()}")
            return median_image
            
        except Exception as e:
            print(f"❌ Error getting satellite image: {e}")
            return None
    
    def download_image(self, image, aoi, filename, scale=30):
        """Download image as GeoTIFF"""
        try:
            print(f"\n💾 Downloading image as {filename}")
            
            # Create output directory
            output_dir = Path("data/debug_downloads")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = output_dir / f"{filename}.tif"
            
            # Use geemap for download
            geemap.download_ee_image(
                image=image,
                filename=str(filepath),
                region=aoi,
                scale=scale,
                crs='EPSG:4326'
            )
            
            print(f"✅ Image saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            # Try alternative export method
            return self.export_to_drive(image, aoi, filename)
    
    def export_to_drive(self, image, aoi, filename):
        """Export to Google Drive as backup"""
        try:
            print(f"🚀 Exporting {filename} to Google Drive...")
            
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=f"debug_{filename}",
                folder='EarthEngine_Debug',
                fileNamePrefix=filename,
                region=aoi,
                scale=30,
                crs='EPSG:4326',
                maxPixels=1e9
            )
            
            task.start()
            
            print(f"✅ Export task started: {task.id}")
            print("📂 Check your Google Drive > EarthEngine_Debug folder")
            print("🌐 Monitor at: https://code.earthengine.google.com/tasks")
            
            return f"drive_export_{task.id}"
            
        except Exception as e:
            print(f"❌ Drive export failed: {e}")
            return None
    
    def visualize_image(self, image, aoi, title="Satellite Image"):
        """Create interactive map visualization"""
        try:
            print(f"\n🗺️ Creating visualization: {title}")
            
            # Create map centered on AOI
            aoi_center = aoi.centroid().coordinates().getInfo()
            lon, lat = aoi_center[0], aoi_center[1]
            
            # Create map
            Map = geemap.Map(center=[lat, lon], zoom=13)
            
            # Add RGB visualization
            if 'red' in image.bandNames().getInfo():
                vis_params = {
                    'bands': ['red', 'green', 'blue'],
                    'min': 0,
                    'max': 0.3,
                    'gamma': 1.4
                }
                Map.addLayer(image, vis_params, f'{title} - RGB')
            
            # Add false color (NIR, Red, Green)
            if 'nir' in image.bandNames().getInfo():
                vis_params_fc = {
                    'bands': ['nir', 'red', 'green'],
                    'min': 0,
                    'max': 0.3,
                    'gamma': 1.4
                }
                Map.addLayer(image, vis_params_fc, f'{title} - False Color')
            
            # Add AOI boundary
            Map.addLayer(aoi, {'color': 'red'}, 'AOI Boundary')
            
            # Save map
            map_file = f"data/debug_downloads/{title.replace(' ', '_').lower()}_map.html"
            Path("data/debug_downloads").mkdir(parents=True, exist_ok=True)
            Map.to_html(map_file)
            
            print(f"✅ Map saved to: {map_file}")
            return Map
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return None
    
    def test_full_workflow(self, location="kolkata"):
        """Test complete workflow"""
        print("🧪 Testing Complete Satellite Data Workflow")
        print("=" * 50)
        
        # 1. Create AOI
        aoi = self.create_test_aoi(location)
        if not aoi:
            return False
        
        # 2. Define time periods
        end_date = datetime.now()
        start_date_recent = end_date - timedelta(days=30)
        start_date_old = end_date - timedelta(days=365)
        end_date_old = start_date_old + timedelta(days=30)
        
        dates = {
            'recent': (start_date_recent.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            'old': (start_date_old.strftime('%Y-%m-%d'), end_date_old.strftime('%Y-%m-%d'))
        }
        
        images = {}
        
        # 3. Get images for both periods
        for period, (start, end) in dates.items():
            print(f"\n{'='*20} {period.upper()} PERIOD {'='*20}")
            image = self.get_satellite_image(aoi, start, end)
            
            if image:
                images[period] = image
                
                # Download image
                filename = f"{location}_{period}_{start.replace('-', '')}"
                downloaded = self.download_image(image, aoi, filename)
                
                # Create visualization
                self.visualize_image(image, aoi, f"{location.title()} {period.title()}")
                
        # 4. Summary
        print(f"\n{'='*20} SUMMARY {'='*20}")
        print(f"✅ Successfully retrieved {len(images)} image periods")
        print(f"📁 Files saved to: data/debug_downloads/")
        
        if len(images) == 2:
            print("✅ Ready for change detection analysis!")
            return True
        else:
            print("⚠️ Need both time periods for change detection")
            return False

def main():
    """Main function to run the debugger"""
    print("🌍 Satellite Data Retrieval Debugger")
    print("=" * 40)
    
    try:
        debugger = SatelliteDataDebugger()
        
        # Test with different locations
        locations = ["kolkata", "bangalore", "delhi"]
        
        print("\nAvailable test locations:")
        for i, loc in enumerate(locations, 1):
            print(f"  {i}. {loc.title()}")
        
        # For now, test with Kolkata
        print("\n🎯 Testing with Kolkata...")
        success = debugger.test_full_workflow("kolkata")
        
        if success:
            print("\n🎉 All tests passed! Your setup is working correctly.")
            print("\n📝 Next steps:")
            print("1. Check the downloaded files in data/debug_downloads/")
            print("2. Open the HTML map files to visualize the data")
            print("3. Use these images for change detection testing")
        else:
            print("\n⚠️ Some tests failed. Check the errors above.")
            
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check Google Earth Engine authentication")
        print("2. Ensure you have proper credentials")
        print("3. Check internet connection")

if __name__ == "__main__":
    main()
