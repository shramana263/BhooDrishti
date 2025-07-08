"use client";
import React from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix for default marker icons in Leaflet with Next.js
const DefaultIcon = L.icon({
  iconUrl: "/location-pin.png",
  shadowUrl: "/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

interface SatelliteMapProps {
  position: [number, number];
}

const SatelliteMap: React.FC<SatelliteMapProps> = ({ position }) => {
  return (
    <MapContainer
      center={position}
      zoom={16}
      style={{ width: "100%", height: "100%" }}
    >
      <TileLayer
        url={`https://api.maptiler.com/maps/satellite/{z}/{x}/{y}.jpg?key=${process.env.NEXT_PUBLIC_MAPTILER_KEY}`}
        attribution='&copy; <a href="https://www.maptiler.com/">MapTiler</a> contributors'
      />
      <Marker position={position} icon={DefaultIcon}>
        <Popup>
          Current Location
          <br />
          Lat: {position[0].toFixed(6)}, Lng: {position[1].toFixed(6)}
        </Popup>
      </Marker>
    </MapContainer>
  );
};

export default SatelliteMap;
