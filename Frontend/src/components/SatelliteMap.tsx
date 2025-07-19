"use client";
import React, { useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import dynamic from "next/dynamic";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Dynamically import DrawingControls to avoid SSR issues
const DrawingControls = dynamic(() => import("./DrawingControls"), { ssr: false });

// Fix for default marker icons in Leaflet with Next.js
const DefaultIcon = L.icon({
  iconUrl: "/location-pin.png",
  shadowUrl: "/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

interface SatelliteMapProps {
  position: [number, number];
  onPolygonCreated?: (coordinates: number[][]) => void;
}

const SatelliteMap: React.FC<SatelliteMapProps> = ({ position, onPolygonCreated }) => {
  const [polygonCoordinates, setPolygonCoordinates] = useState<number[][]>([]);

  const handlePolygonCreated = (coordinates: number[][]) => {
    setPolygonCoordinates(coordinates);
    console.log("New polygon created with coordinates:", coordinates);
    
    // Pass coordinates to parent component if callback provided
    if (onPolygonCreated) {
      onPolygonCreated(coordinates);
    }
  };

  return (
    <div className="relative w-full h-full">
      <MapContainer
        center={position}
        zoom={16}
        style={{ width: "100%", height: "100%" }}
      >
        <TileLayer
          url={`https://api.maptiler.com/maps/satellite/{z}/{x}/{y}.jpg?key=eOgwEnX1eyZsTiqP9t4C`}
          attribution='&copy; <a href="https://www.maptiler.com/">MapTiler</a> contributors'
        />
        <Marker position={position} icon={DefaultIcon}>
          <Popup>
            Current Location
            <br />
            Lat: {position[0].toFixed(6)}, Lng: {position[1].toFixed(6)}
          </Popup>
        </Marker>
        
        {/* Drawing Controls */}
        <DrawingControls onPolygonCreated={handlePolygonCreated} />
      </MapContainer>
      
      {/* Instructions overlay */}
      <div className="absolute top-2 left-2 bg-white bg-opacity-90 p-3 rounded-lg shadow-md max-w-xs z-[1000]">
        <h3 className="text-sm font-semibold text-gray-800 mb-1">Drawing Instructions</h3>
        <p className="text-xs text-gray-600">
          Use the polygon tool (⬟) in the top-right corner to draw your Area of Interest (AOI). 
          Click to start drawing, and double-click to complete the polygon.
        </p>
        {polygonCoordinates.length > 0 && (
          <div className="mt-2 p-2 bg-green-50 rounded border">
            <p className="text-xs text-green-700">
              ✓ Polygon created with {polygonCoordinates.length} points
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SatelliteMap;
