import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    // /api 요청을 백엔드(8000)로 전달. 반드시 http://127.0.0.1:5173 으로 접속. 설정 변경 시 dev 서버 재시작.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyReq", (req) => {
            console.log("[vite proxy] →", req.method, req.url);
          });
        },
      },
    },
  },
});
