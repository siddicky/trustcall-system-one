import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      // The sign-in page is its own entry, so it renders without the app.
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        signin: resolve(import.meta.dirname, "signin.html"),
      },
    },
  },
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
})
