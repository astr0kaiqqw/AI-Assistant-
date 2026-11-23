import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwind from '@tailwindcss/vite'

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/workspace-assets/' : '/',
  plugins: [react(), tailwind()],
  server: {
    proxy: Object.fromEntries(
      ['/chat', '/conversations', '/workspace'].map(path => [path, 'http://127.0.0.1:8000']),
    ),
  },
}))
