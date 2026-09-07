import React, { useState } from 'react';
import DisclaimerBanner from '../components/DisclaimerBanner';
import ImageUploader from '../components/ImageUploader';
import PredictionResult from '../components/PredictionResult';
import UncertaintyGauge from '../components/UncertaintyGauge';
import ExplainabilityViewer from '../components/ExplainabilityViewer';
import ClinicalReportCard from '../components/ClinicalReportCard';
import { predictOralImage } from '../services/api';
import { AlertCircle, Activity } from 'lucide-react';

export default function ScreeningPage() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [mcPasses, setMcPasses] = useState(15);
  const [alpha, setAlpha] = useState(0.5);

  const handleAnalyze = async (file) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const data = await predictOralImage(file, mcPasses, alpha);
      setResult(data);
    } catch (err) {
      console.error('Analysis error:', err);
      setErrorMessage(err.message || 'An error occurred while analyzing the oral image.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setErrorMessage(null);
  };

  return (
    <div>
      <DisclaimerBanner />

      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.02em' }}>
          Oral Lesion Risk Screening
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginTop: '0.2rem' }}>
          Deep learning decision support combining Vision Transformers, Monte Carlo Dropout uncertainty, and XAI Grad-CAM attribution.
        </p>
      </div>

      <div className="screening-grid">
        {/* Left Column: Image Uploader & Controls */}
        <div>
          <ImageUploader
            onAnalyze={handleAnalyze}
            isLoading={isLoading}
            mcPasses={mcPasses}
            setMcPasses={setMcPasses}
            alpha={alpha}
            setAlpha={setAlpha}
            onReset={handleReset}
          />

          {errorMessage && (
            <div style={{
              marginTop: '1.25rem',
              padding: '1rem',
              borderRadius: '8px',
              background: 'rgba(244, 63, 94, 0.15)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: '#fb7185',
              fontSize: '0.85rem',
              display: 'flex',
              gap: '0.6rem',
              alignItems: 'flex-start'
            }}>
              <AlertCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Inference Notice:</strong>
                <p>{errorMessage}</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results & Explainability */}
        <div>
          {result ? (
            <div>
              <PredictionResult result={result} />
              <UncertaintyGauge result={result} />
              <ExplainabilityViewer result={result} />
              <ClinicalReportCard result={result} />
            </div>
          ) : (
            <div className="glass-card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem', color: '#64748b' }}>
              <Activity size={48} style={{ color: '#334155', margin: '0 auto 1rem', display: 'block' }} />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#94a3b8', marginBottom: '0.5rem' }}>
                Awaiting Image Submission
              </h3>
              <p style={{ fontSize: '0.85rem', maxWidth: '380px', margin: '0 auto' }}>
                Upload an oral cavity photograph or select a quick demonstration sample on the left to view the 3-class risk assessment and Grad-CAM attention heatmap.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
