import React, { useState } from 'react';
import { Eye, Layers, Sparkles, Sliders } from 'lucide-react';

export default function ExplainabilityViewer({ result }) {
  const [viewMode, setViewMode] = useState('split'); // 'split' | 'overlay' | 'heatmap'
  const [liveAlpha, setLiveAlpha] = useState(0.55);

  if (!result) return null;

  const { original_image, explanation_heatmap, explanation_overlay, primary_region } = result;

  return (
    <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <span style={{
            fontSize: '0.75rem',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: '#94a3b8',
            fontWeight: 600
          }}>
            Section 4 • Explainable AI (ViT Grad-CAM)
          </span>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc', marginTop: '0.2rem' }}>
            Visual Attention & Token Attribution
          </h3>
        </div>

        {/* View Toggle */}
        <div style={{ display: 'flex', gap: '0.35rem', background: '#0f172a', padding: '0.25rem', borderRadius: '6px' }}>
          <button
            className={`sample-pill ${viewMode === 'split' ? 'active' : ''}`}
            onClick={() => setViewMode('split')}
            style={{ borderColor: viewMode === 'split' ? '#38bdf8' : 'transparent' }}
          >
            Side-by-Side
          </button>
          <button
            className={`sample-pill ${viewMode === 'overlay' ? 'active' : ''}`}
            onClick={() => setViewMode('overlay')}
            style={{ borderColor: viewMode === 'overlay' ? '#38bdf8' : 'transparent' }}
          >
            Live Blend
          </button>
          <button
            className={`sample-pill ${viewMode === 'heatmap' ? 'active' : ''}`}
            onClick={() => setViewMode('heatmap')}
            style={{ borderColor: viewMode === 'heatmap' ? '#38bdf8' : 'transparent' }}
          >
            Heatmap Only
          </button>
        </div>
      </div>

      <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: '#38bdf8' }}>
        <Sparkles size={16} />
        <span>Focal Attention Region: <strong>{primary_region || 'Central mucosal area'}</strong></span>
      </div>

      {/* Viewers */}
      {viewMode === 'split' && (
        <div className="xai-comparison">
          <div className="image-card">
            <span className="image-card-label">Original Oral Cavity</span>
            <img src={original_image} alt="Original Oral" />
          </div>

          <div className="image-card">
            <span className="image-card-label">ViT Grad-CAM Overlay</span>
            <img src={explanation_overlay} alt="Grad-CAM Overlay" />
          </div>
        </div>
      )}

      {viewMode === 'heatmap' && (
        <div style={{ maxWidth: '420px', margin: '1.5rem auto' }} className="image-card">
          <span className="image-card-label">Jet Colormap Heatmap</span>
          <img src={explanation_heatmap} alt="Grad-CAM Heatmap" />
        </div>
      )}

      {viewMode === 'overlay' && (
        <div style={{ margin: '1.5rem 0', textAlign: 'center' }}>
          <div style={{ maxWidth: '420px', margin: '0 auto', position: 'relative', borderRadius: '12px', overflow: 'hidden' }}>
            {/* Background: Original Image */}
            <img
              src={original_image}
              alt="Original"
              style={{ width: '100%', height: 'auto', display: 'block', aspectRatio: '1', objectFit: 'cover' }}
            />
            {/* Foreground: Heatmap with live opacity */}
            <img
              src={explanation_heatmap}
              alt="Heatmap Layer"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                opacity: liveAlpha,
                mixBlendMode: 'screen',
                transition: 'opacity 0.1s ease',
              }}
            />
          </div>

          <div style={{ maxWidth: '380px', margin: '1rem auto 0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#94a3b8', marginBottom: '0.3rem' }}>
              <span>Interactive Layer Opacity:</span>
              <span style={{ color: '#38bdf8', fontWeight: 600 }}>{(liveAlpha * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={liveAlpha}
              onChange={(e) => setLiveAlpha(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#06b6d4' }}
            />
          </div>
        </div>
      )}

      <p style={{
        fontSize: '0.78rem',
        color: '#94a3b8',
        background: 'rgba(15, 23, 42, 0.4)',
        padding: '0.65rem 1rem',
        borderRadius: '8px',
        borderLeft: '3px solid #06b6d4',
        fontStyle: 'italic',
        marginTop: '0.5rem'
      }}>
        "Highlighted regions indicate image areas that influenced the model prediction. They do not represent a medical diagnosis."
      </p>
    </div>
  );
}
