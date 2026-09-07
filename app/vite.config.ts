import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Two proxies for two servers, deliberately. The operational API carries no
// privileged data; the reviewer API carries verdicts and is a separate process
// that is simply not running during agent evaluation.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/review': { target: 'http://127.0.0.1:8001', changeOrigin: true },
    },
  },
})
