import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Use static export to avoid serverless function size issues
  output: "export",
  // Disable image optimization (not available in static export)
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
