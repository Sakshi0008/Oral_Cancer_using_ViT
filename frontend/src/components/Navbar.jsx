import React from 'react';
import { Activity, ShieldAlert, Cpu, BarChart3, Info, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, healthInfo }) {
  const isOnline = healthInfo?.status === 'operational';

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="nav-brand">
          <div className="brand-icon">
            <Activity size={22} />
          </div>
          <div>
            <h1 className="brand-title">OralViT Screening</h1>
            <p className="brand-subtitle">Vision Transformers & Explainable AI (XAI)</p>
          </div>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-btn ${activeTab === 'screening' ? 'active' : ''}`}
            onClick={() => setActiveTab('screening')}
          >
            <Activity size={16} />
            Screening
          </button>

          <button
            className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <BarChart3 size={16} />
            Model Dashboard
          </button>

          <button
            className={`nav-btn ${activeTab === 'architecture' ? 'active' : ''}`}
            onClick={() => setActiveTab('architecture')}
          >
            <Cpu size={16} />
            ViT Architecture
          </button>
        </nav>

        <div className="system-status-pill" title={`Device: ${healthInfo?.device || 'N/A'}`}>
          <div className={`status-dot ${isOnline ? '' : 'offline'}`} />
          <span>{isOnline ? 'Backend Online' : 'Connecting...'}</span>
          {healthInfo?.cuda_available && (
            <span style={{ color: '#38bdf8', borderLeft: '1px solid #334155', paddingLeft: '0.4rem' }}>
              CUDA GPU
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
