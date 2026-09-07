import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sliders, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';

export default function ImageUploader({
  onAnalyze,
  isLoading,
  mcPasses,
  setMcPasses,
  alpha,
  setAlpha,
  onReset
}) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileDetails, setFileDetails] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    setErrorMsg(null);
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
      setErrorMsg('Unsupported format. Please upload a JPEG, PNG, WEBP, or BMP image.');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    // Read image dimensions
    const img = new Image();
    img.onload = () => {
      setFileDetails({
        name: file.name,
        sizeKb: (file.size / 1024).toFixed(1),
        width: img.width,
        height: img.height,
      });
    };
    img.src = objectUrl;
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  // Helper to create synthetic sample images directly in browser canvas for instant demo testing
  const loadSyntheticSample = (category) => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Base mucosal pink tone
    const grad = ctx.createRadialGradient(128, 128, 20, 128, 128, 140);
    grad.addColorStop(0, '#d9777f');
    grad.addColorStop(1, '#b04a55');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 256);

    // Mucosal vascular capillaries
    ctx.strokeStyle = '#8a202d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, 40);
    ctx.bezierCurveTo(70, 90, 110, 60, 150, 110);
    ctx.moveTo(180, 180);
    ctx.bezierCurveTo(140, 150, 170, 110, 210, 80);
    ctx.stroke();

    if (category === 'low-risk') {
      // Low risk: circular aphthous ulcer with erythematous halo
      ctx.fillStyle = '#9e2a36';
      ctx.beginPath();
      ctx.arc(130, 130, 28, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#fef08a';
      ctx.beginPath();
      ctx.arc(130, 130, 20, 0, Math.PI * 2);
      ctx.fill();
    } else if (category === 'high-risk') {
      // High risk: irregular speckled erythroplakic lesion with white hyperkeratosis
      ctx.fillStyle = '#7f1d1d';
      ctx.beginPath();
      ctx.ellipse(135, 125, 45, 32, Math.PI / 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#f8fafc';
      ctx.beginPath();
      ctx.moveTo(120, 115);
      ctx.lineTo(145, 110);
      ctx.lineTo(150, 130);
      ctx.lineTo(130, 135);
      ctx.closePath();
      ctx.fill();

      ctx.beginPath();
      ctx.arc(115, 138, 8, 0, Math.PI * 2);
      ctx.fill();
    }

    canvas.toBlob((blob) => {
      const fileName = `demo_sample_${category}.png`;
      const file = new File([blob], fileName, { type: 'image/png' });
      handleFile(file);
    }, 'image/png');
  };

  const handleStartAnalysis = () => {
    if (!selectedFile) return;
    onAnalyze(selectedFile);
  };

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc' }}>
            Oral Image Upload
          </h2>
          <p style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
            Upload a smartphone or clinical photograph of the oral cavity lesion
          </p>
        </div>

        <button
          className="btn-secondary"
          onClick={() => setShowSettings(!showSettings)}
          title="Inference & Uncertainty Settings"
        >
          <Sliders size={14} />
          {showSettings ? 'Hide Options' : 'Options'}
        </button>
      </div>

      {showSettings && (
        <div style={{
          background: 'rgba(30, 41, 59, 0.7)',
          padding: '1rem',
          borderRadius: '8px',
          marginBottom: '1.25rem',
          border: '1px solid rgba(56, 189, 248, 0.2)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.4rem' }}>
            <span>Monte Carlo Dropout Stochastic Passes: <strong>{mcPasses}</strong></span>
            <span style={{ color: '#38bdf8' }}>{mcPasses >= 15 ? 'High Precision' : 'Fast Preview'}</span>
          </div>
          <input
            type="range"
            min="5"
            max="30"
            step="1"
            value={mcPasses}
            onChange={(e) => setMcPasses(parseInt(e.target.value))}
            style={{ width: '100%', marginBottom: '1rem', accentColor: '#06b6d4' }}
          />

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.4rem' }}>
            <span>Default Grad-CAM Heatmap Blend (Alpha): <strong>{alpha.toFixed(2)}</strong></span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: '#06b6d4' }}
          />
        </div>
      )}

      {/* Drag and Drop Zone */}
      <div
        className={`dropzone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && handleFile(e.target.files[0])}
        />

        {previewUrl ? (
          <div style={{ textAlign: 'center' }}>
            <img
              src={previewUrl}
              alt="Preview"
              style={{
                maxHeight: '220px',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                marginBottom: '0.8rem',
              }}
            />
            {fileDetails && (
              <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                <span style={{ color: '#38bdf8' }}>{fileDetails.name}</span> • {fileDetails.width}×{fileDetails.height}px • {fileDetails.sizeKb} KB
              </div>
            )}
            <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.4rem' }}>
              Click or drag to replace image
            </p>
          </div>
        ) : (
          <div>
            <UploadCloud size={44} style={{ color: '#38bdf8', marginBottom: '0.75rem' }} />
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.35rem' }}>
              Drag & Drop Oral Cavity Photo
            </h3>
            <p style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
              or click to browse from computer (JPEG, PNG, WEBP)
            </p>
          </div>
        )}
      </div>

      {errorMsg && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          color: '#fb7185',
          fontSize: '0.82rem',
          marginTop: '0.75rem'
        }}>
          <AlertCircle size={16} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Quick Starter Demo Samples */}
      <div style={{ marginTop: '1.25rem' }}>
        <p style={{ fontSize: '0.78rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Quick Demonstration Samples:
        </p>
        <div className="sample-pills">
          <button className="sample-pill" onClick={() => loadSyntheticSample('normal')}>
            • Sample Normal Oral Mucosa
          </button>
          <button className="sample-pill" onClick={() => loadSyntheticSample('low-risk')}>
            • Sample Low Risk Lesion
          </button>
          <button className="sample-pill" onClick={() => loadSyntheticSample('high-risk')}>
            • Sample High Risk Lesion
          </button>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ marginTop: '1.5rem', display: 'flex', gap: '0.75rem' }}>
        <button
          className="btn-primary"
          style={{ flex: 1 }}
          disabled={!selectedFile || isLoading}
          onClick={handleStartAnalysis}
        >
          {isLoading ? (
            <>
              <RefreshCw size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
              Running ViT Inference & XAI...
            </>
          ) : (
            <>
              <ImageIcon size={16} />
              Run AI Risk Screening
            </>
          )}
        </button>

        {selectedFile && (
          <button
            className="btn-secondary"
            onClick={() => {
              setSelectedFile(null);
              setPreviewUrl(null);
              setFileDetails(null);
              onReset();
            }}
          >
            Clear
          </button>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
