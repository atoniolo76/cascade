import { renameSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const OUT = resolve(import.meta.dirname, "../landing/onboarding");

/** Vite names the page after its entry; the site serves the folder's index.html at /onboarding/. */
const asIndex: Plugin = {
  name: "onboarding-as-index",
  closeBundle() {
    renameSync(resolve(OUT, "onboarding.html"), resolve(OUT, "index.html"));
  },
};

/** The onboarding demo as static files for the landing site on Vercel. */
export default defineConfig({
  base: "/onboarding/",
  publicDir: false,
  plugins: [react(), tailwindcss(), asIndex],
  build: {
    outDir: OUT,
    emptyOutDir: true,
    rollupOptions: { input: resolve(import.meta.dirname, "onboarding.html") },
  },
});
