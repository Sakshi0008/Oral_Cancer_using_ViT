import React from 'react';
import { Cpu, Layers, GitBranch, Shield, Eye, Compass, HelpCircle } from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function ArchitecturePage() {
  return (
    <div>
      <DisclaimerBanner />

      <div style={{ marginBottom: '2.5rem' }}>
        <h2 style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.02em' }}>
          Vision Transformer Architecture & Explainability
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '0.95rem', marginTop: '0.2rem' }}>
          Technical formulation of ViT-B/16 transfer learning, Monte Carlo Dropout uncertainty sampling, and token-level Grad-CAM attribution.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Card 1: ViT Backbone */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#38bdf8', marginBottom: '0.75rem' }}>
            <Cpu size={20} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
              Vision Transformer (ViT-B/16)
            </h3>
          </div>
          <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: 1.6, marginBottom: '1rem' }}>
            Unlike conventional Convolutional Neural Networks (CNNs) that rely on local receptive fields, the Vision Transformer partitions the 224×224 oral photograph into a 14×14 grid of 16×16 non-overlapping patches.
          </p>
          <ul style={{ fontSize: '0.82rem', color: '#94a3b8', paddingLeft: '1.25rem', lineHeight: 1.8 }}>
            <li><strong>Patch Dimension:</strong> 16×16 pixels (196 spatial tokens + 1 [CLS] token)</li>
            <li><strong>Embedding Dimension:</strong> D = 768 channels</li>
            <li><strong>Transformer Blocks:</strong> 12 Multi-Head Self-Attention layers</li>
            <li><strong>Global Receptive Field:</strong> Captures long-range mucosal dependencies and architectural context</li>
          </ul>
        </div>

        {/* Card 2: MC Dropout Uncertainty */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#fbbf24', marginBottom: '0.75rem' }}>
            <Compass size={20} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
              Monte Carlo Dropout Uncertainty
            </h3>
          </div>
          <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: 1.6, marginBottom: '1rem' }}>
            Standard softmax outputs represent point estimates prone to overconfident misclassifications on unseen out-of-distribution oral lesions. We implement Monte Carlo Dropout in the classification head (p = 0.3).
          </p>
          <ul style={{ fontSize: '0.82rem', color: '#94a3b8', paddingLeft: '1.25rem', lineHeight: 1.8 }}>
            <li><strong>Inference Sampling:</strong> N = 15 stochastic forward passes with dropout active</li>
            <li><strong>Consensus Probability:</strong> Mean softmax distribution p̄ across all passes</li>
            <li><strong>Epistemic Variance:</strong> Measures dispersion across stochastic iterations</li>
            <li><strong>Predictive Entropy:</strong> H(p) normalized between [0, 1]</li>
          </ul>
        </div>

        {/* Card 3: ViT Grad-CAM */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#a78bfa', marginBottom: '0.75rem' }}>
            <Eye size={20} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
              Transformer-Compatible Grad-CAM
            </h3>
          </div>
          <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: 1.6, marginBottom: '1rem' }}>
            Because ViTs do not have standard 2D convolution feature maps, we hook into the output of the final Transformer Encoder LayerNorm.
          </p>
          <ul style={{ fontSize: '0.82rem', color: '#94a3b8', paddingLeft: '1.25rem', lineHeight: 1.8 }}>
            <li><strong>Gradients:</strong> ∂y_c / ∂A_k computed for target risk class c</li>
            <li><strong>Spatial Reconstruction:</strong> Discards [CLS] token and reshapes 196 tokens to a 14×14 2D grid</li>
            <li><strong>Bilinear Upsampling:</strong> Interpolated to full 224×224 resolution</li>
            <li><strong>Alpha Overlay:</strong> Blended over original oral mucosal photograph</li>
          </ul>
        </div>

        {/* Card 4: Clinical 3-Class Taxonomy */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#34d399', marginBottom: '0.75rem' }}>
            <Shield size={20} />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
              Scientifically Grounded Risk Classes
            </h3>
          </div>
          <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: 1.6, marginBottom: '1rem' }}>
            To prevent medically deceptive automated conclusions, the system rejects definitive "Cancer" labels in favor of three defensible risk tiers:
          </p>
          <ul style={{ fontSize: '0.82rem', color: '#94a3b8', paddingLeft: '1.25rem', lineHeight: 1.8 }}>
            <li><strong>Normal:</strong> Baseline intact, healthy oral epithelium</li>
            <li><strong>Low Risk of Malignant Transformation:</strong> Benign / reactive lesions (aphthae, mild lichenoid lesions)</li>
            <li><strong>High Risk of Malignant Transformation:</strong> High-grade dysplasia, leukoplakia, erythroplakia, or suspected OSCC</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
