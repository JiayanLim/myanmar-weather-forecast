import { StrictMode, Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { App } from './App';

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      const err = this.state.error as Error;
      return (
        <div style={{
          position: 'fixed', inset: 0, background: '#0f1117',
          color: '#e2e8f0', display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', padding: '2rem',
          fontFamily: 'system-ui, sans-serif',
        }}>
          <p style={{ color: '#f87171', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Application Error
          </p>
          <p style={{ color: '#94a3b8', fontSize: '0.875rem', maxWidth: '480px', textAlign: 'center' }}>
            {err.message}
          </p>
          <p style={{ color: '#475569', fontSize: '0.75rem', marginTop: '1rem' }}>
            Check the browser console for details. Try refreshing the page.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
