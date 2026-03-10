import React, { useState } from 'react';
import './VerificationForm.css';

function VerificationForm({ onVerify, loading }) {
  const [text, setText] = useState('');
  const [charCount, setCharCount] = useState(0);

  const handleChange = (e) => {
    const value = e.target.value;
    setText(value);
    setCharCount(value.length);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.length >= 300) {
      onVerify(text);
    }
  };

  const isDisabled = text.length < 300 || loading;

  return (
    <div className="verification-form-container">
      <form onSubmit={handleSubmit} className="verification-form">
        <label htmlFor="text-input" className="form-label">
          Введите текст новости для проверки (минимум 300 символов)
        </label>
        <textarea
          id="text-input"
          className="text-input"
          value={text}
          onChange={handleChange}
          placeholder="Вставьте текст новости здесь..."
          rows={10}
          disabled={loading}
        />
        <div className="form-footer">
          <div className="char-count">
            {charCount} / 300 символов
            {charCount < 300 && (
              <span className="char-warning">
                {' '}(нужно еще {300 - charCount})
              </span>
            )}
          </div>
          <button
            type="submit"
            className="verify-button"
            disabled={isDisabled}
          >
            {loading ? 'Проверка...' : 'Проверить'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default VerificationForm;

