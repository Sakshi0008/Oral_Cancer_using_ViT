import React from 'react';
import { FileText, Stethoscope, AlertCircle, ArrowUpRight } from 'lucide-react';

export default function ClinicalReportCard({ result }) {
  if (!result || !result.clinical_summary) return null;

  const { clinical_summary } = result;
  const { urgency, finding_summary, visual_explanation, clinical_recommendation, disclaimer } = clinical_summary;

  const getUrgencyBadge = (urg) => {
    if (urg.includes('High')) {
      return { bg: 'rgba(244, 63, 94, 0.15)', color: '#fb7185', border: 'rgba(244, 63, 94, 0.35)' };
    }
    if (urg.includes('Routine') || urg.includes('Periodic')) {
      return { bg: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', border: 'rgba(245, 158, 11, 0.35)' };
    }
    return { bg: 'rgba(16, 185, 129, 0.15)', color: '#34d399', border: 'rgba(16, 185, 129, 0.35)' };
  };

  const badgeStyle = getUrgencyBadge(urgency || '');

  return (
    <div className="glass-card" style={{ marginBottom: '1.5rem', border: '1px solid rgba(56, 189, 248, 0.25)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Stethoscope size={20} style={{ color: '#38bdf8' }} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
            Clinical Screening Report (Sections 5 & 6)
          </h3>
        </div>

        <span style={{
          fontSize: '0.75rem',
          fontWeight: 700,
          padding: '0.35rem 0.8rem',
          borderRadius: '9999px',
          background: badgeStyle.bg,
          color: badgeStyle.color,
          border: `1px solid ${badgeStyle.border}`,
          textTransform: 'uppercase',
          letterSpacing: '0.04em'
        }}>
          {urgency}
        </span>
      </div>

      {/* Section 5: Clinical-Friendly Explanation */}
      <div style={{ marginBottom: '1.2rem' }}>
        <h4 style={{ fontSize: '0.85rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.35rem' }}>
          Section 5 • Morphologic Finding Summary:
        </h4>
        <p style={{ fontSize: '0.92rem', color: '#e2e8f0', lineHeight: 1.6, background: 'rgba(30, 41, 59, 0.4)', padding: '0.85rem', borderRadius: '8px' }}>
          {finding_summary}
        </p>
      </div>

      <div style={{ marginBottom: '1.2rem' }}>
        <h4 style={{ fontSize: '0.85rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.35rem' }}>
          Visual Attribution Note:
        </h4>
        <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: 1.5 }}>
          {visual_explanation}
        </p>
      </div>

      {/* Section 6: Recommendation */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.75)',
        border: '1px solid rgba(56, 189, 248, 0.2)',
        borderRadius: '10px',
        padding: '1rem',
        marginBottom: '1rem'
      }}>
        <h4 style={{ fontSize: '0.88rem', color: '#38bdf8', fontWeight: 700, marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <ArrowUpRight size={16} />
          Section 6 • Recommended Clinical Action:
        </h4>
        <p style={{ fontSize: '0.9rem', color: '#f1f5f9', lineHeight: 1.6 }}>
          {clinical_recommendation}
        </p>
      </div>

      {/* Mandatory Disclaimer footer */}
      <div style={{
        fontSize: '0.74rem',
        color: '#94a3b8',
        borderTop: '1px solid #334155',
        paddingTop: '0.75rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.4rem'
      }}>
        <AlertCircle size={14} style={{ color: '#f59e0b', flexShrink: 0, marginTop: '2px' }} />
        <span>{disclaimer}</span>
      </div>
    </div>
  );
}
