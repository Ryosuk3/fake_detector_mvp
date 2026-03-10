import React from 'react';
import './ResultDisplay.css';

function ResultDisplay({ result }) {
  const getStatusInfo = (status) => {
    const statusMap = {
      CONFIRMED: {
        label: 'Подтверждено',
        icon: '✅',
        color: '#4caf50',
        bgColor: '#e8f5e9'
      },
      REFUTED: {
        label: 'Опровергнуто',
        icon: '❌',
        color: '#f44336',
        bgColor: '#ffebee'
      },
      PARTIALLY_CONFIRMED: {
        label: 'Частично подтверждено',
        icon: '⚠️',
        color: '#ff9800',
        bgColor: '#fff3e0'
      },
      INSUFFICIENT_DATA: {
        label: 'Недостаточно данных',
        icon: '❓',
        color: '#9e9e9e',
        bgColor: '#f5f5f5'
      }
    };
    return statusMap[status] || statusMap.INSUFFICIENT_DATA;
  };

  const statusInfo = getStatusInfo(result.status);
  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <div className="result-display">
      <div className="result-header">
        <div 
className="result-status" style={{ backgroundColor: statusInfo.bgColor }}>
          <span className="status-icon">{statusInfo.icon}</span>
          <span className="status-label" style={{ color: statusInfo.color }}>
            {statusInfo.label}
          </span>
        </div>
        <div className="result-confidence">
          <span className="confidence-label">Уверенность:</span>
          <span className="confidence-value">{confidencePercent}%</span>
        </div>
      </div>

      {result.warnings && result.warnings.length > 0 && (
        <div className="warnings">
          {result.warnings.map((warning, idx) => (
            <div key={idx} className="warning-item">
              ⚠️ {warning}
            </div>
          ))}
        </div>
      )}

      {result.claims && result.claims.length > 0 && (
        <div className="claims-section">
          <h3>Проверенные утверждения:</h3>
          {result.claims.map((claim, idx) => (
            <div key={idx} className="claim-item">
              <div className="claim-header">
                <span className="claim-text">{claim.claim}</span>
                <span className={`claim-label claim-${claim.label.toLowerCase()}`}>
                  {claim.label === 'ENTAILS' && '✅ Подтверждено'}
                  {claim.label === 'CONTRADICTS' && '❌ Опровергнуто'}
                  {claim.label === 'NEUTRAL' && '➖ Нейтрально'}
                </span>
              </div>
              {claim.evidence && claim.evidence.length > 0 && (
                <div className="claim-evidence">
                  <strong>Источники:</strong>
                  {claim.evidence.map((ev, evIdx) => (
                    <div key={evIdx} className="evidence-item">
                      <a href={ev.url} target="_blank" rel="noopener noreferrer" className="evidence-link">
                        {ev.title || ev.url}
                      </a>
                      {ev.date && <span className="evidence-date">{ev.date}</span>}
                      <div className="evidence-snippet">{ev.snippet}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {result.sources && result.sources.length > 0 && (
        <div className="sources-section">
          <h3>Использованные источники:</h3>
          <div className="sources-list">
            {result.sources.map((source, idx) => (
              <div key={idx} className="source-item">
                <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-link">
                  {source.domain}
                </a>
                {source.date && <span className="source-date">{source.date}</span>}
                <span className="source-trust">
                  Доверие: {Math.round(source.trust_level * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ResultDisplay;

