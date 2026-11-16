import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 650,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three", "three/examples/jsm/controls/OrbitControls.js"],
          react: ["react", "react-dom"]
        }
      }
    }
  },
  server: {
    port: 5173,
    strictPort: false
  }
});
