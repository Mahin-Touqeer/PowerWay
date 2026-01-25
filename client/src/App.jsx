import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import SidebarBox from "./pages/Sidebar/SidebarBox.jsx";
import Weather from "./components/Weather.jsx";
import Dashboard from "./pages/Dashboard/Dashboard.jsx";
import Load from "./pages/Load/Load.jsx";
import Graph from "./pages/Graph/Graph.jsx";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Navigate to="home" />} />
          <Route path="home" element={<Dashboard />} />
          <Route path="load" element={<Load />} />
          <Route path="graphs" element={<Graph />} />
        </Route>
        {/* <Route path="/weather" element={<Weather />} /> */}
      </Routes>
    </BrowserRouter>
  );
}

function MainLayout() {
  return (
    <>
      <SidebarBox />
    </>
  );
}
function Home() {
  return <h1>Home</h1>;
}

export default App;
