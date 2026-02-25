/**
 * 개발 시 Vite 프록시가 동작하지 않는 환경에서 백엔드(8000)로 직접 요청.
 * 프로덕션 빌드에서는 '' (같은 origin).
 */
export const API_BASE = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";
