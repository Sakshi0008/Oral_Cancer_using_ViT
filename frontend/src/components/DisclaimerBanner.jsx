import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner">
      <AlertTriangle size={24} className="disclaimer-icon" />
      <div className="disclaimer-text">
        <strong>Academic Research Screening & Decision-Support Notice:</strong>
        <p>
          This system provides AI-assisted preliminary oral lesion risk screening using Vision Transformers and is NOT a medical diagnosis.
          It does not replace clinical histopathology, biopsy, or evaluation by a qualified dental, oral medicine, or oncology specialist.
        </p>
      </div>
    </div>
  );
}
