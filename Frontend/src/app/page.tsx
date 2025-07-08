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
        <h1 className="text-[20px] font-bold">Area of Interest(AOI) manager</h1>
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
              <div className="w-full h-20 text-xs font-light border-[1px] border-gray-300 rounded-lg p-2 mt-1 flex items-center justify-center">
                No Coordinates added
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
              className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-lg"
            >
              Create AOI
            </button>
          </form>
        </div>
      </div>
      <div className="min-h-screen w-9/12 ">
        <div style={{ width: "100%", height: "100vh" }}>
          <SatelliteMap position={position} />
        </div>
      </div>
    </div>
  );
}
