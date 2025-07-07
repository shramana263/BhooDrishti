import React from "react";
import Link from "next/link";
import { GoDownload } from "react-icons/go";

export default function Navbar() {
  return (
    <div className="flex justify-between max-w-7xl mx-auto py-2 items-center border-b-[1px]">
      <h1 className="text-2xl">BhooDristi</h1>
      <div className="flex gap-3 items-center">
        <Link href={"/"}>Dashboard</Link>
        <Link href={"/"}>Alerts</Link>
        <button className="flex items-center gap-1 border-2 rounded-md border-gray-400 py-[2px] px-[6px]">
            <GoDownload />
            <p>Export</p>
        </button>
      </div>
    </div>
  );
}
