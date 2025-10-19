import React, { useState, KeyboardEvent } from 'react';
import { Send, Trash2, Loader2 } from 'lucide-react';
import './MultiModalApp.css';

const API_URL = "http://localhost:8000";

interface Result {
  intent: string;
  type: 'text' | 'image' | 'audio' | 'translation';
  result: string;
  original?: string;
  file_path?: string;
  image_data?: string;
  audio_data?: string;
}

interface Example {
  label: string;
  text: string;
}

const MultiModalApp: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  const examples: Example[] = [
    { label: 'Text', text: 'Tell me a story about a robot learning to paint' },
    { label: 'Image', text: 'Create an image of a futuristic city at night' },
    { label: 'Audio', text: 'Read this message: Welcome to our AI service' },
    { label: 'Translation', text: 'Translate to Marathi: How are you today?' }
  ];

  const processPrompt = async (): Promise<void> => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim() })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process request');
      }

      const data: Result = await response.json();
      console.log('Response data:', data);
      setResult(data);
    } catch (err) {
      console.error('Error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = (): void => {
    setPrompt('');
    setResult(null);
    setError(null);
  };

  const fillExample = (text: string): void => {
    setPrompt(text);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      processPrompt();
    }
  };

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement>): void => {
    console.error('Image load error');
    if (result?.file_path) {
      e.currentTarget.src = `${API_URL}/${result.file_path.replace(/\\/g, '/')}`;
    }
  };

  const handleAudioError = (e: React.SyntheticEvent<HTMLAudioElement>): void => {
    console.error('Audio load error from base64');
    if (result?.file_path) {
      e.currentTarget.src = `${API_URL}/${result.file_path.replace(/\\/g, '/')}`;
    }
  };

  return (
    <div className="app-container">
      <div className="main-container">
        {/* Header */}
        <div className="header">
          <h1 className="header-title">AI Multi-Modal Assistant</h1>
          <p className="header-subtitle">
            Generate text, images, audio, and translations using advanced AI models
          </p>
        </div>

        {/* Main Content */}
        <div className="chat-container">
          {/* Input Section */}
          <div className="input-section">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe what you'd like to create... Try 'Draw a sunset', 'Translate hello to Marathi', or 'Read this aloud'"
              className="input-textarea"
            />
            
            <div className="button-group">
              <button
                onClick={processPrompt}
                disabled={loading}
                className="btn-primary"
              >
                {loading ? (
                  <>
                    <Loader2 className="icon spin-animation" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Send className="icon" />
                    Generate
                  </>
                )}
              </button>
              <button onClick={clearAll} className="btn-secondary">
                <Trash2 className="icon" />
                Clear
              </button>
            </div>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="loading-container">
              <Loader2 className="loading-spinner" />
              <span className="loading-text">Processing your request...</span>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="error-container">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="result-container fade-in">
              <div className="intent-badge">
                <span className="intent-dot"></span>
                {result.intent}
              </div>
              
              <div className="result-content">
                {/* Text Result */}
                {result.type === 'text' && (
                  <div className="result-text">{result.result}</div>
                )}

                {/* Translation Result */}
                {result.type === 'translation' && (
                  <>
                    <div className="success-message">
                      ✓ Translation completed
                    </div>
                    <div className="translation-result">
                      <div className="translation-item">
                        <strong>Original:</strong>
                        <div className="translation-text">{result.original}</div>
                      </div>
                      <div className="translation-item">
                        <strong>Marathi:</strong>
                        <div className="translation-text marathi">{result.result}</div>
                      </div>
                    </div>
                  </>
                )}

                {/* Image Result */}
                {result.type === 'image' && (
                  <>
                    <div className="success-message">
                      ✓ Image generated successfully
                    </div>
                    {result.image_data ? (
                      <img
                        src={result.image_data}
                        alt="Generated"
                        className="result-image"
                        onLoad={() => console.log('Image loaded from base64')}
                        onError={handleImageError}
                      />
                    ) : (
                      <img
                        src={`${API_URL}/${result.file_path?.replace(/\\/g, '/')}`}
                        alt="Generated"
                        className="result-image"
                        onError={() => console.error('Image failed to load from path')}
                      />
                    )}
                  </>
                )}

                {/* Audio Result */}
                {result.type === 'audio' && (
                  <>
                    <div className="success-message">
                      ✓ Audio generated successfully
                    </div>
                    <div className="audio-description">{result.result}</div>
                    {result.audio_data ? (
                      <audio
                        controls
                        autoPlay
                        className="result-audio"
                        onLoadedData={() => console.log('Audio loaded from base64')}
                        onError={handleAudioError}
                      >
                        <source src={result.audio_data} type="audio/mpeg" />
                        Your browser does not support audio playback.
                      </audio>
                    ) : (
                      <audio
                        controls
                        autoPlay
                        className="result-audio"
                        onError={() => console.error('Audio failed to load from path')}
                      >
                        <source src={`${API_URL}/${result.file_path?.replace(/\\/g, '/')}`} type="audio/mpeg" />
                        Your browser does not support audio playback.
                      </audio>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Examples */}
          <div className="examples-section">
            <h3 className="examples-title">Try these examples</h3>
            <div className="examples-grid">
              {examples.map((example, index) => (
                <button
                  key={index}
                  onClick={() => fillExample(example.text)}
                  className="example-card"
                >
                  <span className="example-label">{example.label}</span>
                  <span className="example-text">{example.text}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MultiModalApp;