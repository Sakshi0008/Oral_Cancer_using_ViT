import React from 'react';
import { HelpCircle, Percent, Compass } from 'lucide-react';

export default function UncertaintyGauge({ result }) {
  if (!result) return null;

  const { confidence_pct, uncertainty_level, uncertainty_score } = result;

  const getUncertaintyColor = (level) => {
    if (level === 'Low') return '#10b981';
    if (level === 'Moderate') return '#f59e0b';
    return '#f43f5e';
  };

  const uncColor = getUncertaintyColor(uncertainty_level);

  return (
    <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
      <span style={{
        fontSize: '0.75rem',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        color: '#94a3b8',
        fontWeight: 600
      }}>
        Section 3 • Model Confidence & Uncertainty
      </span>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '1rem',
        marginTop: '0.9rem',
        marginBottom: '1rem'
      }}>
        {/* Confidence Card */}
        <div style={{
          background: 'rgba(30, 41, 59, 0.5)',
          padding: '1rem',
          borderRadius: '10px',
          border: '1px solid rgba(56, 189, 248, 0.2)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#38bdf8', fontSize: '0.8rem', marginBottom: '0.2rem' }}>
            <Percent size={14} />
            <span>Model Confidence</span>
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc', fontFamily: 'var(--font-mono)' }}>
            {confidence_pct}
          </div>
          <p style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '0.2rem' }}>
            Mean probability across stochastic MC passes
          </p>
        </div>

        {/* Uncertainty Card */}
        <div style={{
          background: 'rgba(30, 41, 59, 0.5)',
          padding: '1rem',
          borderRadius: '10px',
          border: `1px solid ${uncColor}44`
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: uncColor, fontSize: '0.8rem', marginBottom: '0.2rem' }}>
            <Compass size={14} />
            <span>Uncertainty Level</span>
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: uncColor, fontFamily: 'var(--font-mono)' }}>
            {uncertainty_level}
          </div>
          <p style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '0.2rem' }}>
            Epistemic variance score: {uncertainty_score}
          </p>
        </div>
      </div>

      <div style={{
        background: 'rgba(2, 6, 23, 0.5)',
        padding: '0.65rem 0.85rem',
        borderRadius: '6px',
        border: '1px solid rgba(255, 255, 255, 0.05)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '0.74rem',
        color: '#94a3b8'
      }}>
        <HelpCircle size={15} style={{ color: '#38bdf8', flexShrink: 0 }} />
        <span>
          Confidence and uncertainty are derived via Monte Carlo Dropout sampling and reflect statistical model stability.
          They must not be construed as clinical or diagnostic certainty.
        </span>
      </div>
    </div>
  );
}
