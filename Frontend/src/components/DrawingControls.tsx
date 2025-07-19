"use client";
import React from "react";
import { FeatureGroup } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";
import "leaflet-draw/dist/leaflet.draw.css";

interface DrawingControlsProps {
  onPolygonCreated: (coordinates: number[][]) => void;
}

const DrawingControls: React.FC<DrawingControlsProps> = ({ onPolygonCreated }) => {
  const handleCreated = (e: { layerType: string; layer: { getLatLngs: () => Array<Array<{ lat: number; lng: number }>> } }) => {
    const { layerType, layer } = e;
    
    if (layerType === "polygon") {
      // Get the coordinates of the polygon
      const coordinates = layer.getLatLngs()[0].map((latlng: { lat: number; lng: number }) => [
        latlng.lat,
        latlng.lng
      ]);
      
      console.log("Polygon coordinates:", coordinates);
      onPolygonCreated(coordinates);
    }
  };

  return (
    <FeatureGroup>
      <EditControl
        position="topright"
        onCreated={handleCreated}
        draw={{
          rectangle: false,
          polygon: {
            allowIntersection: false,
            drawError: {
              color: "#e1e100",
              message: "<strong>Error:</strong> Shape edges cannot cross!"
            },
            shapeOptions: {
              color: "#97009c",
              weight: 2,
              fillOpacity: 0.2
            }
          },
          circle: false,
          circlemarker: false,
          marker: false,
          polyline: false,
        }}
        edit={{
          featureGroup: undefined,
          remove: true
        }}
      />
    </FeatureGroup>
  );
};

export default DrawingControls;
