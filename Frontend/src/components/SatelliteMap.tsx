"use client";
import React, { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import MapboxDraw from "mapbox-gl-draw";
import "maplibre-gl/dist/maplibre-gl.css";
import "mapbox-gl-draw/dist/mapbox-gl-draw.css";

const MAPTILER_KEY = "eOgwEnX1eyZsTiqP9t4C"; // Replace with your key

export default function SatelliteMap() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    if (mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current!,
      style: `https://api.maptiler.com/maps/satellite/style.json?key=${MAPTILER_KEY}`,
      center: [88.3639, 22.5726],
      zoom: 12,
    });

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: {
        polygon: true,
        trash: true,
      },
    });
    map.addControl(draw, "top-right");

    map.on("draw.create", (e) => {
      const coords = e.features[0].geometry.coordinates;
      console.log("AOI Polygon Coordinates:", coords);
    });

    mapRef.current = map;

    return () => {
      map.remove();
    };
  }, []);

  return (
    <div
      ref={mapContainer}
      style={{ width: "100%", height: "100vh", borderRadius: "8px" }}
    />
  );
}