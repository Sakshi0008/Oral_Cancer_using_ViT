import React from 'react';
import { ShieldCheck, AlertTriangle, AlertOctagon } from 'lucide-react';

export default function PredictionResult({ result }) {
  if (!result) return null;

  const { predicted_class, probabilities } = result;

  const getBadgeStyle = (riskClass) => {
    if (riskClass === 'Normal') {
      return {
        className: 'risk-badge normal',
        icon: <ShieldCheck size={26} />,
        barColor: '#10b981',
      };
    }
    if (riskClass === 'Low Risk of Malignant Transformation') {
      return {
        className: 'risk-badge low-risk',
        icon: <AlertTriangle size={26} />,
        barColor: '#f59e0b',
      };
    }
    return {
      className: 'risk-badge high-risk',
      icon: <AlertOctagon size={26} />,
      barColor: '#f43f5e',
    };
  };

  const badgeInfo = getBadgeStyle(predicted_class);

  return (
    <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
      <span style={{
        fontSize: '0.75rem',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        color: '#94a3b8',
        fontWeight: 600
      }}>
        Section 2 • Screening Classification
      </span>

      <div style={{ marginTop: '0.8rem', marginBottom: '1.5rem' }}>
        <div className={badgeInfo.className}>
          {badgeInfo.icon}
          <span>{predicted_class}</span>
        </div>
      </div>

      {/* Probability Distribution */}
      <div>
        <h4 style={{ fontSize: '0.85rem', color: '#cbd5e1', marginBottom: '0.85rem', fontWeight: 600 }}>
          Softmax Class Probabilities (Across MC Passes):
        </h4>

        {probabilities && Object.entries(probabilities).map(([className, prob]) => {
          const pct = (prob * 100).toFixed(1);
          let barColor = '#64748b';
          if (className === 'Normal') barColor = '#10b981';
          else if (className === 'Low Risk of Malignant Transformation') barColor = '#f59e0b';
          else if (className === 'High Risk of Malignant Transformation') barColor = '#f43f5e';

          const isTop = className === predicted_class;

          return (
            <div key={className} className="prob-item">
              <div className="prob-header">
                <span style={{ fontWeight: isTop ? 700 : 400, color: isTop ? '#f8fafc' : '#94a3b8' }}>
                  {className}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: isTop ? barColor : '#cbd5e1' }}>
                  {pct}%
                </span>
              </div>
              <div className="prob-bar-bg">
                <div
                  className="prob-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: barColor,
                    boxShadow: isTop ? `0 0 10px ${barColor}88` : 'none',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
