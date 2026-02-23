import { Routes, Route, NavLink } from "react-router-dom";
import LipstickEffect from "./pages/LipstickEffect";
import Recommendation from "./pages/Recommendation";
import MacroDashboard from "./pages/MacroDashboard";

function App() {
  return (
    <div className="app">
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
      </nav>
      <main className="main">
        <Routes>
          <Route path="/" element={<LipstickEffect />} />
          <Route path="/recommendation" element={<Recommendation />} />
          <Route path="/macro" element={<MacroDashboard />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
