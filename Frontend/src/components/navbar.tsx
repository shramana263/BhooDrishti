import React from "react";
import { Download, Satellite } from "lucide-react";
import Link from "next/link";

export default function Navbar() {
  return (
    <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Link href={"/"} className="flex items-center space-x-2">
                <Satellite className="h-8 w-8 text-blue-600" />
                <span className="text-xl font-bold text-gray-900">BhooDrishti</span>
              </Link>
              <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">Map View</span>
            </div>
            <div className="flex items-center space-x-4">
              <Link href={"/dashboard"} className="text-gray-600 hover:text-gray-900">Dashboard</Link>
              <button className="text-gray-600 hover:text-gray-900">Alerts</button>
              <button className="flex items-center space-x-1 text-gray-600 hover:text-gray-900">
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>
            </div>
          </div>
        </div>
      </header>
  );
}
