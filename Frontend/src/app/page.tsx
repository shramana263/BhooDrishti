"use client";
import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import the SatelliteMap component to ensure client-side rendering
const SatelliteMap = dynamic(() => import("@/components/SatelliteMap"), { ssr: false });

export default function Page() {
  const [position, setPosition] = useState<[number, number]>([
    22.5726,
    88.3639,
  ]); // Default to Kolkata
  const [aoiCoordinates, setAoiCoordinates] = useState<number[][]>([]);

  useEffect(() => {
    if (typeof window !== "undefined" && "geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setPosition([latitude, longitude]);
        },
        (error) => {
          console.error("Geolocation error:", error);
          // Fallback to Kolkata
          setPosition([22.58468291, 88.34724883]);
        }
      );
    } else {
      // Geolocation not supported
      setPosition([22.5726, 88.3639]);
      console.log("Hello else part");
    }
  }, []);

  return (
    <div className="px-2 flex w-full ">
      <div className="left w-3/12  min-h-screen ">
        <h1 className="text-[20px] font-bold">Area of Interest (AOI) Manager</h1>
        <div className="border-[1px] border-gray-300 rounded-lg p-4 mt-4">
          <h2 className="text-2xl font-bold">Create New AOI</h2>
          <p className="text-xs font-light">Define a new area to monitor</p>

          <form>
            <div className="mt-4">
              <label className="block text-sm font-medium">AOI Name</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg p-2 mt-1"
                placeholder="Enter AOI name"
              />
            </div>
            <div className="mt-4">
              <label className="block text-sm font-medium">Description</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg p-2 mt-1"
                placeholder="Enter description"
                rows={3}
              ></textarea>
            </div>
            <div className="mt-4">
              <label className="block text-sm font-medium">Coordinates</label>
              <div className="w-full min-h-20 text-xs font-light border-[1px] border-gray-300 rounded-lg p-2 mt-1">
                {aoiCoordinates.length > 0 ? (
                  <div>
                    <p className="font-medium text-green-600 mb-2">
                      Polygon with {aoiCoordinates.length} points:
                    </p>
                    <div className="max-h-32 overflow-y-auto">
                      {aoiCoordinates.map((coord, index) => (
                        <div key={index} className="text-xs mb-1">
                          Point {index + 1}: [{coord[0].toFixed(6)}, {coord[1].toFixed(6)}]
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-16">
                    <span className="text-gray-500">Draw a polygon on the map to add coordinates</span>
                  </div>
                )}
              </div>
            </div>
            <div className="mt-4">
              <label className="block text-sm font-medium mb-2">
                Change type of monitor
              </label>
              <div className="flex flex-col gap-2">
                <label className="flex items-center gap-2">
                  <input type="checkbox" name="illegal_building" />
                  Illegal Building
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" name="deforestation" />
                  Deforestation
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" name="water_body_change" />
                  Water Body Change
                </label>
              </div>
            </div>

            <button
              type="submit"
              className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors"
              disabled={aoiCoordinates.length === 0}
            >
              Create AOI
            </button>
            
            {aoiCoordinates.length === 0 && (
              <p className="text-xs text-gray-500 mt-2">
                Please draw a polygon on the map to enable AOI creation
              </p>
            )}
          </form>
        </div>
      </div>
      <div className="min-h-screen w-9/12 ">
        <div style={{ width: "100%", height: "100vh" }}>
          <SatelliteMap position={position} onPolygonCreated={setAoiCoordinates} />
        </div>
      </div>
    </div>
  );
}
