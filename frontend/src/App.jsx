import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ScreeningPage from './pages/ScreeningPage';
import DashboardPage from './pages/DashboardPage';
import ArchitecturePage from './pages/ArchitecturePage';
import { checkHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('screening');
  const [healthInfo, setHealthInfo] = useState(null);

  useEffect(() => {
    const fetchHealth = async () => {
      const data = await checkHealth();
      setHealthInfo(data);
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} healthInfo={healthInfo} />

      <main className="main-content">
        {activeTab === 'screening' && <ScreeningPage />}
        {activeTab === 'dashboard' && <DashboardPage />}
        {activeTab === 'architecture' && <ArchitecturePage />}
      </main>

      <footer className="footer">
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <strong>Oral Cancer Screening Decision-Support System</strong>
            <p style={{ color: '#64748b', fontSize: '0.75rem', marginTop: '0.2rem' }}>
              B.Tech CSE (AIML) Major Project Prototype • Vision Transformers (ViT-B/16) with Explainable AI
            </p>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', maxWidth: '420px', textAlign: 'right' }}>
            Academic research prototype. Not approved as a medical device. Professional medical evaluation is required for clinical diagnosis.
          </div>
        </div>
      </footer>
    </div>
  );
}
