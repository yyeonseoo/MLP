import { useState, useEffect } from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import LipstickEffect from "./pages/LipstickEffect";
import Recommendation from "./pages/Recommendation";
import MacroDashboard from "./pages/MacroDashboard";
import Forecast from "./pages/Forecast";

function App() {
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  useEffect(() => {
    fetch("/api/health")
      .then(async (r) => {
        if (!r.ok) return false;
        const d = await r.json();
        return (d as { status?: string }).status === "ok";
      })
      .then(setApiOk)
      .catch(() => setApiOk(false));
  }, []);

  return (
    <div className="app">
      {apiOk === false && (
        <div
          style={{
            padding: "0.5rem 1rem",
            background: "#7f1d1d",
            color: "#fecaca",
            fontSize: "0.85rem",
            textAlign: "center",
          }}
        >
          API 연결 실패. 서버 확인: 터미널에서{" "}
          <code style={{ background: "#450a0a", padding: "0.2rem 0.4rem", borderRadius: 4 }}>
            PYTHONPATH=. uvicorn api.app:app --host 0.0.0.0 --port 8000
          </code>{" "}
          실행 후 새로고침.
        </div>
      )}
      {apiOk === true && (
        <div
          style={{
            padding: "0.25rem 1rem",
            background: "#14532d",
            color: "#bbf7d0",
            fontSize: "0.8rem",
            textAlign: "center",
          }}
        >
          API 연결됨
        </div>
      )}
      <nav className="nav">
        <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
          립스틱 효과 분석
        </NavLink>
        <NavLink to="/recommendation" className={({ isActive }) => (isActive ? "active" : "")}>
          추천 시스템
        </NavLink>
        <NavLink to="/macro" className={({ isActive }) => (isActive ? "active" : "")}>
          거시경제 대시보드
        </NavLink>
        <NavLink to="/forecast" className={({ isActive }) => (isActive ? "active" : "")}>
          매출 전망
        </NavLink>
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<LipstickEffect />} />
          <Route path="/recommendation" element={<Recommendation />} />
          <Route path="/macro" element={<MacroDashboard />} />
          <Route path="/forecast" element={<Forecast />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
