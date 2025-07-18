#!/usr/bin/env python3
"""
Polygon Coordinate Helper
========================

This script helps you:
1. Define custom polygon coordinates for any location
2. Convert different coordinate formats
3. Validate polygon coordinates
4. Get satellite data for custom areas

Usage:
    python polygon_helper.py
"""

import json
import geemap
from typing import List, Tuple, Dict

class PolygonCoordinateHelper:
    """Helper to create and manage polygon coordinates"""
    
    def __init__(self):
        pass
    
    def create_polygon_from_bounds(self, north: float, south: float, 
                                 east: float, west: float) -> List[List[float]]:
        """Create polygon from bounding box coordinates"""
        coords = [
            [west, south],   # Bottom-left
            [east, south],   # Bottom-right  
            [east, north],   # Top-right
            [west, north],   # Top-left
            [west, south]    # Close polygon
        ]
        
        print(f"📐 Created polygon from bounds:")
        print(f"   North: {north}, South: {south}")
        print(f"   East: {east}, West: {west}")
        print(f"   Coordinates: {coords}")
        
        return coords
    
    def create_polygon_from_center(self, lat: float, lon: float, 
                                 size_km: float = 2.0) -> List[List[float]]:
        """Create square polygon around center point"""
        # Approximate degree offset (varies with latitude)
        lat_offset = size_km / 111.0  # 1 degree ≈ 111 km
        lon_offset = size_km / (111.0 * abs(lat / 90.0))  # Adjust for latitude
        
        coords = [
            [lon - lon_offset, lat - lat_offset],  # Bottom-left
            [lon + lon_offset, lat - lat_offset],  # Bottom-right
            [lon + lon_offset, lat + lat_offset],  # Top-right
            [lon - lon_offset, lat + lat_offset],  # Top-left
            [lon - lon_offset, lat - lat_offset]   # Close polygon
        ]
        
        print(f"📍 Created {size_km}km square around ({lat}, {lon})")
        print(f"   Coordinates: {coords}")
        
        return coords
    
    def validate_coordinates(self, coords: List[List[float]]) -> bool:
        """Validate polygon coordinates"""
        try:
            if len(coords) < 4:
                print("❌ Polygon needs at least 4 points")
                return False
            
            # Check if polygon is closed
            if coords[0] != coords[-1]:
                print("⚠️ Polygon not closed, fixing...")
                coords.append(coords[0])
            
            # Check coordinate ranges
            for i, (lon, lat) in enumerate(coords):
                if not (-180 <= lon <= 180):
                    print(f"❌ Invalid longitude at point {i}: {lon}")
                    return False
                if not (-90 <= lat <= 90):
                    print(f"❌ Invalid latitude at point {i}: {lat}")
                    return False
            
            print("✅ Coordinates are valid")
            return True
            
        except Exception as e:
            print(f"❌ Error validating coordinates: {e}")
            return False
    
    def create_geojson(self, coords: List[List[float]], name: str = "AOI") -> Dict:
        """Create GeoJSON from coordinates"""
        geojson = {
            "type": "Feature",
            "properties": {
                "name": name,
                "description": f"Area of Interest: {name}"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        }
        
        print(f"📄 Created GeoJSON for {name}")
        return geojson
    
    def save_coordinates(self, coords: List[List[float]], filename: str):
        """Save coordinates to file"""
        try:
            # Save as GeoJSON
            geojson = self.create_geojson(coords, filename)
            
            with open(f"data/{filename}.geojson", 'w') as f:
                json.dump(geojson, f, indent=2)
            
            # Save as simple coordinates
            with open(f"data/{filename}_coords.json", 'w') as f:
                json.dump({"coordinates": coords}, f, indent=2)
            
            print(f"💾 Saved coordinates to:")
            print(f"   - data/{filename}.geojson")
            print(f"   - data/{filename}_coords.json")
            
        except Exception as e:
            print(f"❌ Error saving coordinates: {e}")
    
    def get_famous_locations(self) -> Dict[str, Dict]:
        """Get coordinates for famous locations worldwide"""
        locations = {
            # Indian Cities
            "mumbai": {
                "coords": self.create_polygon_from_center(19.0760, 72.8777, 3.0),
                "description": "Mumbai, India"
            },
            "delhi": {
                "coords": self.create_polygon_from_center(28.6139, 77.2090, 3.0),
                "description": "Delhi, India"
            },
            "bangalore": {
                "coords": self.create_polygon_from_center(12.9716, 77.5946, 3.0),
                "description": "Bangalore, India"
            },
            "kolkata": {
                "coords": self.create_polygon_from_center(22.5726, 88.3639, 3.0),
                "description": "Kolkata, India"
            },
            "chennai": {
                "coords": self.create_polygon_from_center(13.0827, 80.2707, 3.0),
                "description": "Chennai, India"
            },
            
            # International Cities
            "london": {
                "coords": self.create_polygon_from_center(51.5074, -0.1278, 3.0),
                "description": "London, UK"
            },
            "paris": {
                "coords": self.create_polygon_from_center(48.8566, 2.3522, 3.0),
                "description": "Paris, France"
            },
            "tokyo": {
                "coords": self.create_polygon_from_center(35.6762, 139.6503, 3.0),
                "description": "Tokyo, Japan"
            },
            "new_york": {
                "coords": self.create_polygon_from_center(40.7128, -74.0060, 3.0),
                "description": "New York, USA"
            },
            
            # Natural Areas
            "amazon_rainforest": {
                "coords": self.create_polygon_from_center(-3.4653, -62.2159, 10.0),
                "description": "Amazon Rainforest, Brazil"
            },
            "sahara_desert": {
                "coords": self.create_polygon_from_center(23.8859, 2.5085, 10.0),
                "description": "Sahara Desert, Algeria"
            }
        }
        
        return locations
    
    def interactive_location_creator(self):
        """Interactive tool to create custom locations"""
        print("\n🎯 Interactive Location Creator")
        print("=" * 30)
        
        print("\nChoose creation method:")
        print("1. From center point and size")
        print("2. From bounding box (North, South, East, West)")
        print("3. Choose from famous locations")
        print("4. Manual coordinate entry")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                return self._create_from_center()
            elif choice == "2":
                return self._create_from_bounds()
            elif choice == "3":
                return self._choose_famous_location()
            elif choice == "4":
                return self._manual_coordinate_entry()
            else:
                print("❌ Invalid choice")
                return None
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None
    
    def _create_from_center(self):
        """Create polygon from center point"""
        try:
            print("\n📍 Enter center coordinates:")
            lat = float(input("Latitude: "))
            lon = float(input("Longitude: "))
            size = float(input("Size in km (default 2.0): ") or 2.0)
            name = input("Location name: ").strip() or "custom_location"
            
            coords = self.create_polygon_from_center(lat, lon, size)
            
            if self.validate_coordinates(coords):
                self.save_coordinates(coords, name)
                return coords, name
            return None
            
        except ValueError:
            print("❌ Invalid numeric input")
            return None
    
    def _create_from_bounds(self):
        """Create polygon from bounding box"""
        try:
            print("\n📐 Enter bounding box coordinates:")
            north = float(input("North latitude: "))
            south = float(input("South latitude: "))
            east = float(input("East longitude: "))
            west = float(input("West longitude: "))
            name = input("Location name: ").strip() or "custom_bounds"
            
            coords = self.create_polygon_from_bounds(north, south, east, west)
            
            if self.validate_coordinates(coords):
                self.save_coordinates(coords, name)
                return coords, name
            return None
            
        except ValueError:
            print("❌ Invalid numeric input")
            return None
    
    def _choose_famous_location(self):
        """Choose from predefined famous locations"""
        locations = self.get_famous_locations()
        
        print("\n🌍 Famous Locations:")
        for i, (key, info) in enumerate(locations.items(), 1):
            print(f"  {i:2d}. {info['description']}")
        
        try:
            choice = int(input(f"\nChoose location (1-{len(locations)}): "))
            location_key = list(locations.keys())[choice - 1]
            location_info = locations[location_key]
            
            coords = location_info['coords']
            name = location_key
            
            print(f"\n✅ Selected: {location_info['description']}")
            self.save_coordinates(coords, name)
            return coords, name
            
        except (ValueError, IndexError):
            print("❌ Invalid choice")
            return None
    
    def _manual_coordinate_entry(self):
        """Manual coordinate entry"""
        print("\n✏️ Manual Coordinate Entry")
        print("Enter coordinates as: longitude,latitude")
        print("Enter 'done' when finished (minimum 4 points)")
        
        coords = []
        while True:
            try:
                point_input = input(f"Point {len(coords) + 1}: ").strip()
                
                if point_input.lower() == 'done':
                    if len(coords) >= 3:
                        # Close polygon
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        break
                    else:
                        print("❌ Need at least 3 points")
                        continue
                
                lon, lat = map(float, point_input.split(','))
                coords.append([lon, lat])
                print(f"✅ Added point: ({lon}, {lat})")
                
            except ValueError:
                print("❌ Invalid format. Use: longitude,latitude")
            except KeyboardInterrupt:
                print("\n❌ Cancelled")
                return None
        
        name = input("Location name: ").strip() or "manual_polygon"
        
        if self.validate_coordinates(coords):
            self.save_coordinates(coords, name)
            return coords, name
        return None

def main():
    """Main function"""
    print("🌍 Polygon Coordinate Helper")
    print("=" * 30)
    
    helper = PolygonCoordinateHelper()
    
    # Show some examples
    print("\n📝 Example: Creating polygon for Mumbai")
    mumbai_coords = helper.create_polygon_from_center(19.0760, 72.8777, 2.0)
    helper.save_coordinates(mumbai_coords, "mumbai_example")
    
    print("\n📝 Example: Creating polygon from bounding box")
    bbox_coords = helper.create_polygon_from_bounds(
        north=28.7041, south=28.4041, 
        east=77.3500, west=77.0500
    )
    helper.save_coordinates(bbox_coords, "delhi_bbox_example")
    
    # Interactive mode
    print("\n" + "="*50)
    result = helper.interactive_location_creator()
    
    if result:
        coords, name = result
        print(f"\n🎉 Successfully created polygon for '{name}'")
        print("📁 Files saved to data/ directory")
        print("\n📋 Use these coordinates in your satellite data script:")
        print(f"   coords = {coords}")
    else:
        print("\n❌ No polygon created")

if __name__ == "__main__":
    main()
