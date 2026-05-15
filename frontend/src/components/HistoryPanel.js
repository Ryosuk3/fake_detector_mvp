import React from 'react';
import './HistoryPanel.css';

const STATUS_LABELS = {
  CONFIRMED: 'Подтверждено',
  REFUTED: 'Опровергнуто',
  PARTIALLY_CONFIRMED: 'Частично',
  INSUFFICIENT_DATA: 'Недостаточно данных',
};

function formatDate(value) {
  if (!value) {
    return '';
  }

  return new Intl.DateTimeFormat('ru-RU', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value));
}

function HistoryPanel({ items, loading, error, selectedId, onRefresh, onSelect }) {
  return (
    <aside className="history-panel">
      <div className="history-header">
        <div>
          <p className="history-kicker">Архив</p>
          <h2>История проверок</h2>
        </div>
        <button className="history-refresh" type="button" onClick={onRefresh} disabled={loading}>
          {loading ? '...' : 'Обновить'}
        </button>
      </div>

      {error && <div className="history-error">{error}</div>}

      {!loading && items.length === 0 && !error && (
        <div className="history-empty">История появится после первой проверки.</div>
      )}

      <div className="history-list">
        {items.map((item) => {
          const status = item.status || item.result?.status || 'INSUFFICIENT_DATA';
          const confidence = item.confidence ?? item.result?.confidence ?? 0;
          const isSelected = String(item.id) === String(selectedId);

          return (
            <button
              type="button"
              className={`history-item ${isSelected ? 'selected' : ''}`}
              key={item.id}
              onClick={() => onSelect(item)}
            >
              <div className="history-item-top">
                <span className={`history-status status-${status.toLowerCase()}`}>
                  {STATUS_LABELS[status] || status}
                </span>
                <span className="history-date">{formatDate(item.created_at)}</span>
              </div>
              <p>{item.text_preview}</p>
              <div className="history-meta">
                <span>{Math.round(confidence * 100)}% уверенность</span>
                <span>#{item.id}</span>
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

export default HistoryPanel;
