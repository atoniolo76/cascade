import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../cascade/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": { target: process.env.CASCADE_API ?? "http://localhost:8000", changeOrigin: true },
    },
  },
});
