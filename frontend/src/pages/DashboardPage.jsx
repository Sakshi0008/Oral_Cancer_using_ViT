import React, { useState, useEffect } from 'react';
import { getMetrics, getStaticOutputUrl } from '../services/api';
import { BarChart2, CheckCircle, RefreshCw, AlertCircle, Award, Target, Layers } from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function DashboardPage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedImgModal, setSelectedImgModal] = useState(null);

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err) {
      console.error('Error fetching metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  return (
    <div>
      <DisclaimerBanner />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.02em' }}>
            Model Performance & Research Evaluation
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginTop: '0.2rem' }}>
            Quantitative evaluation metrics on held-out test splits. No fabricated values; derived strictly from model evaluation.
          </p>
        </div>

        <button className="btn-secondary" onClick={fetchMetrics} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh Metrics
        </button>
      </div>

      {loading ? (
        <div className="glass-card" style={{ textAlign: 'center', padding: '3rem', color: '#94a3b8' }}>
          <RefreshCw size={32} style={{ animation: 'spin 1s linear infinite', margin: '0 auto 1rem', color: '#38bdf8' }} />
          <p>Querying model evaluation metrics...</p>
        </div>
      ) : !metrics || !metrics.available ? (
        <div className="glass-card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
          <BarChart2 size={52} style={{ color: '#475569', margin: '0 auto 1.25rem' }} />
          <h3 style={{ fontSize: '1.25rem', color: '#f8fafc', fontWeight: 700, marginBottom: '0.5rem' }}>
            Model Evaluation Will Appear After Training
          </h3>
          <p style={{ color: '#94a3b8', maxWidth: '520px', margin: '0 auto 1.5rem', fontSize: '0.9rem' }}>
            {metrics?.message || 'No evaluation metrics found in outputs/metrics.json.'}
          </p>

          <div style={{
            background: '#090d16',
            border: '1px solid #334155',
            borderRadius: '8px',
            padding: '1rem',
            maxWidth: '480px',
            margin: '0 auto',
            textAlign: 'left',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.82rem',
            color: '#38bdf8'
          }}>
            <p style={{ color: '#94a3b8', marginBottom: '0.5rem' }}># Step 1: Train the Vision Transformer:</p>
            <p style={{ color: '#f8fafc' }}>python ml/train.py</p>
            <p style={{ color: '#94a3b8', margin: '0.75rem 0 0.5rem' }}># Step 2: Evaluate on the test split:</p>
            <p style={{ color: '#f8fafc' }}>python ml/evaluate.py</p>
          </div>
        </div>
      ) : (
        <div>
          {/* Key Stat Cards */}
          <div className="metrics-grid">
            <div className="metric-stat-card">
              <div className="metric-label">Overall Test Accuracy</div>
              <div className="metric-value" style={{ color: '#34d399' }}>
                {metrics.accuracy_pct || `${(metrics.accuracy * 100).toFixed(1)}%`}
              </div>
              <div className="metric-sub">{metrics.total_test_samples} Test Samples Evaluated</div>
            </div>

            <div className="metric-stat-card">
              <div className="metric-label">Macro F1-Score</div>
              <div className="metric-value" style={{ color: '#38bdf8' }}>
                {metrics.macro_f1?.toFixed(4) || 'N/A'}
              </div>
              <div className="metric-sub">Unweighted Class Average</div>
            </div>

            <div className="metric-stat-card">
              <div className="metric-label">Weighted F1-Score</div>
              <div className="metric-value" style={{ color: '#a78bfa' }}>
                {metrics.weighted_f1?.toFixed(4) || 'N/A'}
              </div>
              <div className="metric-sub">Support-Weighted Metric</div>
            </div>

            <div className="metric-stat-card">
              <div className="metric-label">Macro Precision / Recall</div>
              <div className="metric-value" style={{ fontSize: '1.35rem', marginTop: '0.3rem', color: '#f8fafc' }}>
                {metrics.macro_precision?.toFixed(3)} / {metrics.macro_recall?.toFixed(3)}
              </div>
              <div className="metric-sub">Balanced Detection Tradeoff</div>
            </div>
          </div>

          {/* Per-Class Metrics Table */}
          {metrics.per_class_metrics && (
            <div className="glass-card" style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem' }}>
                Per-Class Performance Metrics
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #334155', textAlign: 'left', color: '#94a3b8' }}>
                      <th style={{ padding: '0.75rem 1rem' }}>Risk Category</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Precision</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Recall</th>
                      <th style={{ padding: '0.75rem 1rem' }}>F1-Score</th>
                      <th style={{ padding: '0.75rem 1rem' }}>Test Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.per_class_metrics).map(([cName, vals]) => (
                      <tr key={cName} style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
                        <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: '#f8fafc' }}>{cName}</td>
                        <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)' }}>{(vals.precision * 100).toFixed(1)}%</td>
                        <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)' }}>{(vals.recall * 100).toFixed(1)}%</td>
                        <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', color: '#38bdf8', fontWeight: 600 }}>{vals.f1_score.toFixed(3)}</td>
                        <td style={{ padding: '0.85rem 1rem', color: '#94a3b8' }}>{vals.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Visualization Plots: Confusion Matrix & Training Curves */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
            {metrics.confusion_matrix_url && (
              <div className="glass-card">
                <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.75rem' }}>
                  Test Split Confusion Matrix
                </h4>
                <div
                  className="image-card"
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSelectedImgModal(getStaticOutputUrl(metrics.confusion_matrix_url))}
                >
                  <img src={getStaticOutputUrl(metrics.confusion_matrix_url)} alt="Confusion Matrix" />
                </div>
                <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                  Click image to expand view
                </p>
              </div>
            )}

            {metrics.training_curves_url && (
              <div className="glass-card">
                <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.75rem' }}>
                  Training & Validation Curves
                </h4>
                <div
                  className="image-card"
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSelectedImgModal(getStaticOutputUrl(metrics.training_curves_url))}
                >
                  <img src={getStaticOutputUrl(metrics.training_curves_url)} alt="Training Curves" />
                </div>
                <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                  Click image to expand view
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Image Zoom Modal */}
      {selectedImgModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.85)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '2rem',
          }}
          onClick={() => setSelectedImgModal(null)}
        >
          <div style={{ maxWidth: '900px', width: '100%', textAlign: 'center' }}>
            <img
              src={selectedImgModal}
              alt="Expanded Plot"
              style={{ maxHeight: '85vh', maxWidth: '100%', borderRadius: '12px', border: '1px solid #38bdf8' }}
            />
            <p style={{ color: '#cbd5e1', fontSize: '0.85rem', marginTop: '0.75rem' }}>Click anywhere to close</p>
          </div>
        </div>
      )}
    </div>
  );
}
