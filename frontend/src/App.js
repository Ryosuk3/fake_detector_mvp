import React, { useState } from 'react';
import './App.css';
import VerificationForm from './components/VerificationForm';
import ResultDisplay from './components/ResultDisplay';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleVerify = async (text) => {
    setLoading(true);
    setError(null);
    setResult(null);

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
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔍 Fake Detector</h1>
        <p>Проверка достоверности информации в новостных текстах</p>
      </header>
      
      <main className="App-main">
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
      </main>
      
      <footer className="App-footer">
        <p>Система использует ML модели для проверки фактов по доверенным источникам</p>
      </footer>
    </div>
  );
}

export default App;

