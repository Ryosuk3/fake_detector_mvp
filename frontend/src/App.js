import React, { useCallback, useEffect, useState } from 'react';
import './App.css';
import VerificationForm from './components/VerificationForm';
import ResultDisplay from './components/ResultDisplay';
import HistoryPanel from './components/HistoryPanel';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8200';

function App() {
  const [result, setResult] = useState(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);

    try {
      const response = await fetch(`${API_URL}/history?limit=30`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Не удалось загрузить историю проверок');
      }

      const data = await response.json();
      setHistory(data);
    } catch (err) {
      setHistoryError(err.message);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const handleVerify = async (text) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedHistoryId(null);

    try {
      const response = await fetch(`${API_URL}/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ошибка при проверке текста');
      }

      const data = await response.json();
      setResult(data);
      setSelectedHistoryId(data.request_id || null);
      loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectHistory = (item) => {
    if (!item.result) {
      return;
    }

    setError(null);
    setResult(item.result);
    setSelectedHistoryId(String(item.id));
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="brand-mark">ДН</div>
        <div>
          <p className="eyebrow">Проверка новостных текстов</p>
          <h1>ДизинфНЕТ</h1>
          <p>Сопоставляем утверждения с доверенными источниками и показываем найденные расхождения.</p>
        </div>
      </header>
      
      <main className="App-main">
        <section className="workspace">
          <div className="primary-column">
            <VerificationForm
              onVerify={handleVerify}
              loading={loading}
            />

            {error && (
              <div className="error-message">
                <strong>Ошибка:</strong> {error}
              </div>
            )}

            {result && <ResultDisplay result={result} />}
          </div>

          <HistoryPanel
            items={history}
            loading={historyLoading}
            error={historyError}
            selectedId={selectedHistoryId}
            onRefresh={loadHistory}
            onSelect={handleSelectHistory}
          />
        </section>
      </main>
      
      <footer className="App-footer">
        <p>ДизинфНЕТ использует ML-модели, поиск по источникам и NLI-верификацию для предварительной проверки фактов.</p>
      </footer>
    </div>
  );
}

export default App;

